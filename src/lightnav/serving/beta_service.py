"""Trained Beta trajectories over the existing MuJoCo/robot WebSocket protocol."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from lightnav.inference.beta_model import uses_frozen_continuous_backbone
from lightnav.inference.config import InferenceConfig
from lightnav.inference.policies import NavigationPolicy
from lightnav.inference.samples import (
    build_continuous_traj_sample,
    build_vln_traj_sample,
    prompt_only_inputs,
)
from lightnav.serving.batcher import MicroBatchScheduler
from lightnav.serving.tracking_service import BatchedTrackingService, TrackingPrediction
from lightnav.tracking import TrackingAgent

logger = logging.getLogger(__name__)


class BetaSession(NavigationPolicy):
    def reset(self, instruction=""):
        super().reset(instruction)
        self.prediction_step = 0
        self.generation = getattr(self, "generation", 0) + 1
        self.completed_prediction = None


class BetaTrackingService(BatchedTrackingService):
    """Serialize HF/FM calls; frame history and random state belong to each client."""

    warmup_strict = True

    def __init__(self, model, bundle, *, num_history_frames, fm_steps=None,
                 seed=42, max_new_tokens=64, rvq_bundle=None, aspect_mode="stretch",
                 stop_mode=None, stop_threshold=0.5):
        self.model = model
        self.bundle = bundle
        self.num_history_frames = num_history_frames
        self.fm_steps = fm_steps
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.fm_only = uses_frozen_continuous_backbone(model.config.training_mode)
        if stop_mode is None:
            stop_mode = "learned" if model.config.training_mode == "fm_stop" else "manual"
        if stop_mode not in {"manual", "rvq", "learned"}:
            raise ValueError("Beta stop_mode must be manual, rvq or learned")
        if stop_mode == "learned" and (
            model.config.training_mode != "fm_stop" or getattr(model, "stop_head", None) is None
        ):
            raise ValueError("learned STOP requires an fm_stop checkpoint with trained StopHead weights")
        if not math.isfinite(stop_threshold) or not 0 <= stop_threshold <= 1:
            raise ValueError("stop_threshold must be finite and between 0 and 1")
        self.stop_mode = stop_mode
        self.stop_threshold = float(stop_threshold)
        self.engine = SimpleNamespace(bundle=bundle, aspect_mode=aspect_mode,
                                      reset_episode_state=lambda: None)
        self.decoder = None
        if not self.fm_only or stop_mode == "rvq":
            if rvq_bundle is None:
                raise ValueError("Beta token STOP requires an RVQ action tokenizer")
            self.decoder = TrackingAgent(self.engine, rvq_bundle=rvq_bundle)
        self.scheduler = MicroBatchScheduler(self._infer_batch, max_batch_size=1, max_wait_ms=0)

    def make_session(self, client_id=None):
        session = BetaSession(self.engine, num_history_frames=self.num_history_frames)
        session.client_id = client_id
        session.reset()
        return session

    @torch.no_grad()
    def _infer_batch(self, requests):
        results = []
        for request in requests:
            try:
                results.append(self._predict_one(request))
            except Exception as exc:
                results.append(exc)
        return results

    def _predict_one(self, request):
        start = time.monotonic()
        session = request.session
        generation = session.generation
        if session.completed_prediction is not None:
            return replace(session.completed_prediction, timings_ms={
                "batch_size": 1.0, "queue_wait_ms": (start - request.submitted_at) * 1000,
                "batch_total_ms": 0.0,
            })
        builder = build_continuous_traj_sample if self.fm_only else build_vln_traj_sample
        sample = builder(session._get_video_tensor(), session.instruction,
                         list(session._history_frame_ids), self.bundle)
        processed = self.bundle.data_processor.process_sample(
            sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        inputs = prompt_only_inputs(processed)
        generator = torch.Generator(device=self.model.expert.step_embedding.device)
        generator.manual_seed((self.seed + session.prediction_step) % (2**63 - 1))
        stop_score = None
        stop_source = ""
        if self.stop_mode == "learned":
            prediction = self.model.predict_continuous(
                inputs, stop_threshold=self.stop_threshold,
                num_steps=self.fm_steps, generator=generator,
            )
            waypoints = prediction.waypoints
            stop = bool(prediction.stop[0].item())
            stop_score = float(prediction.stop_scores[0].item())
            if not math.isfinite(stop_score):
                raise RuntimeError("StopHead produced a non-finite score")
            stop_source = "learned_head" if stop else ""
            raw_text = ""
        elif self.fm_only:
            # Use the frozen backbone's original prompt for STOP only. Motion
            # still uses the separately encoded continuous training prompt.
            stop = self.stop_mode == "rvq" and self._rvq_requests_stop(session)
            raw_text = ""
            if stop:
                stop_source = "rvq_stop"
                waypoints = torch.zeros(1, self.model.config.horizon, 3)
            else:
                context = self.model.encode_context(inputs)
                _, waypoints = self.model.sample_actions(
                    context, num_steps=self.fm_steps, generator=generator,
                )
        else:
            prediction = self.model.predict(
                inputs, tokenizer=self.bundle.tokenizer, action_decoder=self.decoder,
                max_new_tokens=self.max_new_tokens, num_steps=self.fm_steps,
                generator=generator, action_output="fm",
            )
            waypoints = prediction.waypoints
            stop, raw_text = bool(prediction.stop[0].item()), prediction.texts[0]
            if stop:
                stop_source = "rvq_stop"
        trajectory = waypoints[0].float().cpu().numpy().copy()
        if trajectory.shape != (self.model.config.horizon, 3) or not np.isfinite(trajectory).all():
            raise RuntimeError("Beta produced an invalid trajectory")
        if stop:
            trajectory.fill(0)
        prediction = TrackingPrediction(
            waypoints=trajectory, stop=stop, visible=None, traj_id=None, tpos_id=None,
            raw_text=raw_text, stop_score=stop_score,
            stop_threshold=self.stop_threshold if self.stop_mode == "learned" else None,
            stop_source=stop_source, timings_ms={
                "batch_size": 1.0,
                "queue_wait_ms": (start - request.submitted_at) * 1000,
                "batch_total_ms": (time.monotonic() - start) * 1000,
            },
        )
        if generation != session.generation:
            raise RuntimeError("Beta session reset during inference; discarded stale prediction")
        session.prediction_step += 1
        if stop:
            session.completed_prediction = prediction
        return prediction

    def _rvq_requests_stop(self, session):
        sample = build_vln_traj_sample(
            session._get_video_tensor(), session.instruction,
            list(session._history_frame_ids), self.bundle,
        )
        processed = self.bundle.data_processor.process_sample(
            sample, add_generation_prompt=True, validate_video_shapes=False,
        )
        tokenizer = self.bundle.tokenizer
        with self.model._hf_inputs(prompt_only_inputs(processed)) as inputs:
            sequences = self.model.vlm.generate(
                **inputs, use_cache=True, do_sample=False, num_beams=1,
                num_return_sequences=1, return_dict_in_generate=False,
                output_hidden_states=False, output_attentions=False,
                max_new_tokens=self.max_new_tokens, min_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            tokens = sequences[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        decoded, _ = self.decoder.decode_waypoints(text)
        # Malformed tokens propagate as an inference error, not "goal reached".
        return not np.any(decoded)


def build_beta_service(args):
    from lightnav.inference.beta_model import load_beta_checkpoint

    if args.backend != "hf" or args.task != "vln":
        raise ValueError("Beta serving requires --backend hf --task vln")
    config = InferenceConfig(
        model_path=args.model_path, backend="hf", device=args.device,
        num_history_frames=args.num_history_frames, pool_spatial=args.pool_spatial,
        aspect_mode=args.aspect_mode,
    )
    model, bundle, path, metadata = load_beta_checkpoint(config, args.beta_checkpoint)
    rvq_bundle = None
    if not uses_frozen_continuous_backbone(model.config.training_mode) or args.beta_stop_mode == "rvq":
        from lightnav.traj_vocab import load_rvq_bundle

        if not args.action_tokenizer_bundle:
            raise ValueError("CE+FM Beta requires --action_tokenizer_bundle")
        rvq_bundle = load_rvq_bundle(args.action_tokenizer_bundle, args.horizon,
                                     num_frames=0, load_cluster_ids=False)
    args.horizon = model.config.horizon
    logger.info("Beta checkpoint=%s step=%s mode=%s horizon=%d", path,
                metadata.get("step"), model.config.training_mode, args.horizon)
    if uses_frozen_continuous_backbone(model.config.training_mode):
        logger.info("FM-only STOP mode: %s (rvq uses the frozen LightNav STOP decoder)",
                    args.beta_stop_mode)
    return BetaTrackingService(
        model, bundle, num_history_frames=args.num_history_frames or bundle.num_history_frames,
        fm_steps=args.beta_fm_steps, seed=args.beta_seed,
        max_new_tokens=max(64, args.max_new_tokens), rvq_bundle=rvq_bundle,
        aspect_mode=args.aspect_mode,
        stop_mode=args.beta_stop_mode,
        stop_threshold=args.beta_stop_threshold,
    ), bundle
