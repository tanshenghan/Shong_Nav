"""Beta FM serving: continuous prompt, session isolation and wire trajectories."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightnav.serving import beta_service, ws_server
from lightnav.serving.tracking_service import _PredictRequest


class FakeBeta:
    config = SimpleNamespace(training_mode="fm_only", horizon=30)
    expert = SimpleNamespace(step_embedding=torch.zeros(1))

    def encode_context(self, inputs):
        assert not torch.is_grad_enabled()
        assert inputs["input_ids"].tolist() == [[1, 2]]
        assert "labels" not in inputs
        return inputs

    def sample_actions(self, context, *, num_steps, generator):
        assert num_steps == 4
        return None, torch.rand(1, 30, 3, generator=generator)


def make_service(monkeypatch, model=None, **kwargs):
    samples = []

    def build(video, instruction, frame_ids, bundle):
        samples.append((instruction, frame_ids))
        return {"continuous": True}

    def process(sample, **kwargs):
        assert sample == {"continuous": True}
        return {"input_ids": torch.tensor([1, 2, 99]),
                "labels": torch.tensor([-100, -100, 99])}

    monkeypatch.setattr(beta_service, "build_continuous_traj_sample", build)
    bundle = SimpleNamespace(video_size=(32, 32),
                             data_processor=SimpleNamespace(process_sample=process))
    service = beta_service.BetaTrackingService(
        model or FakeBeta(), bundle, num_history_frames=2, fm_steps=4,
        **kwargs,
    )
    return service, samples


@pytest.mark.asyncio
async def test_fm_wire_horizon_and_session_rng_reset(monkeypatch):
    service, samples = make_service(monkeypatch)
    a, b = service.make_session("a"), service.make_session("b")
    frame = np.zeros((32, 32, 3), np.uint8)
    a.reset("walk forward")
    b.reset("turn left")
    a.observe(frame)
    b.observe(frame)
    await service.start()
    try:
        first = await service.predict(a)
        other = await service.predict(b)
        second = await service.predict(a)
        np.testing.assert_array_equal(first.waypoints, other.waypoints)
        assert not np.array_equal(first.waypoints, second.waypoints)
        a.reset("walk forward")
        a.observe(frame)
        replay = await service.predict(a)
        np.testing.assert_array_equal(first.waypoints, replay.waypoints)
    finally:
        await service.stop()
    assert samples[:2] == [("walk forward", [0]), ("turn left", [0])]
    wire = ws_server._success_response(first, 1, 12.0, a, frame)["data"]
    assert len(wire["actions"]["actions"]) == 30
    assert wire["stop"] is False and wire["visible"] is None
    assert "pointing" not in wire


@pytest.mark.asyncio
async def test_empty_history_rejected(monkeypatch):
    service, _ = make_service(monkeypatch)
    with pytest.raises(RuntimeError, match="empty observation"):
        await service.predict(service.make_session())


def test_invalid_prediction_isolated(monkeypatch):
    class InvalidBeta(FakeBeta):
        def sample_actions(self, *args, **kwargs):
            return None, torch.full((1, 30, 3), float("nan"))

    service, _ = make_service(monkeypatch, InvalidBeta())
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    result = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
    assert isinstance(result, RuntimeError)
    assert session.prediction_step == 0


def test_beta_cli_requires_hf_vln_and_no_rvq_for_fm_only(monkeypatch):
    from lightnav.inference import beta_model

    monkeypatch.setattr(beta_model, "resolve_beta_checkpoint", lambda p: (
        "/committed/step-3699", {"beta_config": {"training_mode": "fm_only", "horizon": 30}},
    ))
    parser = ws_server._build_parser()
    base = ["--model_path", "/base", "--beta_checkpoint", "/beta"]
    with pytest.raises(ValueError, match="requires --backend hf --task vln"):
        ws_server.resolve_decoder_args(parser.parse_args(base))
    args = parser.parse_args(base + ["--backend", "hf", "--task", "vln"])
    ws_server.resolve_decoder_args(args)
    assert args.horizon == 30 and args.action_tokenizer_bundle is None
    args.beta_fm_steps = 0
    with pytest.raises(ValueError, match="must be positive"):
        ws_server.resolve_decoder_args(args)


@pytest.mark.asyncio
async def test_beta_warmup_failure_must_not_report_ready(monkeypatch):
    service, _ = make_service(monkeypatch)

    async def broken_predict(session):
        raise ValueError("bad Beta inputs")

    monkeypatch.setattr(service, "predict", broken_predict)
    with pytest.raises(ValueError, match="bad Beta inputs"):
        await ws_server._warmup(service, service.bundle)


def test_ce_fm_preserves_model_stop_and_trajectory(monkeypatch):
    class TokenBeta(FakeBeta):
        config = SimpleNamespace(training_mode="ce_fm", horizon=30)

        def predict(self, inputs, **kwargs):
            assert kwargs["action_output"] == "fm"
            return SimpleNamespace(waypoints=torch.zeros(1, 30, 3),
                                   stop=torch.tensor([True]), texts=["<stop>"])

    # The decoder itself is covered by the existing RVQ tests.
    monkeypatch.setattr(beta_service, "TrackingAgent", lambda *a, **kw: object())
    model = TokenBeta()
    service, _ = make_service(monkeypatch)  # reuse the processor fixture
    service = beta_service.BetaTrackingService(
        model, service.bundle, num_history_frames=2, rvq_bundle=object(),
    )
    service.bundle.tokenizer = object()
    monkeypatch.setattr(beta_service, "build_vln_traj_sample",
                        lambda *a: {"continuous": True})
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    pred = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
    assert pred.stop is True and pred.raw_text == "<stop>"
    assert pred.waypoints.shape == (30, 3)


@pytest.mark.parametrize("requests_stop", [True, False])
def test_fm_only_rvq_gate_keeps_fm_motion_or_stops(monkeypatch, requests_stop):
    service, _ = make_service(monkeypatch)
    service.stop_mode = "rvq"
    monkeypatch.setattr(service, "_rvq_requests_stop", lambda s: requests_stop)
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    pred = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
    assert pred.stop is requests_stop
    assert bool(np.any(pred.waypoints)) is not requests_stop


def test_rvq_stop_uses_a_separate_token_prompt(monkeypatch):
    from contextlib import contextmanager

    service, _ = make_service(monkeypatch)
    monkeypatch.setattr(beta_service, "build_vln_traj_sample", lambda *a: {"continuous": True})

    @contextmanager
    def hf_inputs(inputs):
        yield inputs

    service.model._hf_inputs = hf_inputs
    service.model.vlm = SimpleNamespace(generate=lambda **kw: torch.tensor([[1, 2, 7, 8, 9]]))
    service.bundle.tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=3,
        decode=lambda tokens, **kw: str(tokens.tolist()),
    )
    service.decoder = SimpleNamespace(
        decode_waypoints=lambda text: (np.zeros((10, 3)), text),
    )
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    assert service._rvq_requests_stop(session)
    service.decoder.decode_waypoints = lambda text: (np.ones((10, 3)), text)
    assert not service._rvq_requests_stop(session)


class FakeStopBeta(FakeBeta):
    config = SimpleNamespace(training_mode="fm_stop", horizon=30)
    stop_head = object()

    def __init__(self, scores):
        self.scores = iter(scores)
        self.calls = 0

    def encode_context(self, inputs):
        raise AssertionError("learned STOP must share predict_continuous encoding")

    def predict_continuous(self, inputs, *, stop_threshold, num_steps, generator):
        assert not torch.is_grad_enabled()
        assert inputs["input_ids"].tolist() == [[1, 2]] and "labels" not in inputs
        assert num_steps == 4 and generator is not None
        self.calls += 1
        score = next(self.scores)
        return SimpleNamespace(
            stop=torch.tensor([score >= stop_threshold]), stop_scores=torch.tensor([score]),
            # A terminal flag must take precedence over any returned trajectory.
            waypoints=torch.ones(1, 30, 3),
        )


@pytest.mark.asyncio
async def test_learned_stop_single_prediction_latches_per_session_until_reset(monkeypatch):
    model = FakeStopBeta([0.2, 0.8, 0.1, 0.1])
    service, samples = make_service(monkeypatch, model, stop_threshold=0.7)
    assert service.stop_mode == "learned" and service.decoder is None
    a, b = service.make_session("a"), service.make_session("b")
    frame = np.zeros((32, 32, 3), np.uint8)
    for session in (a, b):
        session.observe(frame)
    await service.start()
    try:
        moving = await service.predict(a)
        assert not moving.stop and np.any(moving.waypoints)
        stopped = await service.predict(a)
        assert stopped.stop and not np.any(stopped.waypoints)
        assert stopped.stop_source == "learned_head"
        for _ in range(2):
            again = await service.predict(a)
            assert again.stop and not np.any(again.waypoints)
        assert model.calls == 2 and len(samples) == 2
        assert not (await service.predict(b)).stop
        a.reset("new task")
        a.observe(frame)
        assert not (await service.predict(a)).stop
        assert model.calls == 4
    finally:
        await service.stop()
    wire = ws_server._success_response(stopped, 1, 1.0, a, frame)["data"]
    assert wire["stop_score"] == pytest.approx(0.8)
    assert wire["stop_threshold"] == 0.7 and wire["stop_source"] == "learned_head"


def test_learned_continue_zero_motion_is_not_completion(monkeypatch):
    model = FakeStopBeta([0.1, 0.1])
    original = model.predict_continuous

    def predict(*args, **kwargs):
        prediction = original(*args, **kwargs)
        prediction.waypoints.zero_()
        return prediction

    model.predict_continuous = predict
    service, _ = make_service(monkeypatch, model)
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    for _ in range(2):
        result = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
        assert not result.stop and result.stop_source == ""
    assert model.calls == 2 and session.completed_prediction is None


def test_old_completion_cannot_latch_a_reset_session(monkeypatch):
    model = FakeStopBeta([0.9])
    service, _ = make_service(monkeypatch, model)
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    original = model.predict_continuous

    def predict(*args, **kwargs):
        session.reset("new task")
        return original(*args, **kwargs)

    model.predict_continuous = predict
    result = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
    assert isinstance(result, RuntimeError) and "stale prediction" in str(result)
    assert session.completed_prediction is None and session.prediction_step == 0


def test_nonfinite_stop_score_is_an_error_not_completion(monkeypatch):
    service, _ = make_service(monkeypatch, FakeStopBeta([float("nan")]))
    session = service.make_session()
    session.observe(np.zeros((32, 32, 3), np.uint8))
    result = service._infer_batch([_PredictRequest(session, time.monotonic())])[0]
    assert isinstance(result, RuntimeError)
    assert session.completed_prediction is None and session.prediction_step == 0


def test_learned_stop_rejects_missing_head_and_bad_threshold(monkeypatch):
    with pytest.raises(ValueError, match="trained StopHead"):
        make_service(monkeypatch, stop_mode="learned")
    model = FakeStopBeta([])
    model.stop_head = None
    with pytest.raises(ValueError, match="trained StopHead"):
        make_service(monkeypatch, model)
    for threshold in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="stop_threshold"):
            make_service(monkeypatch, FakeStopBeta([]), stop_threshold=threshold)


def test_learned_cli_defaults_and_legacy_checkpoint_rejection(monkeypatch):
    from lightnav.inference import beta_model

    config = {"training_mode": "fm_stop", "horizon": 30}
    monkeypatch.setattr(beta_model, "resolve_beta_checkpoint", lambda p: (
        "/committed/step-1", {"beta_config": config},
    ))
    parser = ws_server._build_parser()
    base = ["--model_path", "/base", "--beta_checkpoint", "/beta", "--backend", "hf", "--task", "vln"]
    args = ws_server.resolve_decoder_args(parser.parse_args(base))
    assert args.beta_stop_mode == "learned" and args.beta_stop_threshold == 0.5
    assert args.horizon == 30 and args.action_tokenizer_bundle is None
    config["training_mode"] = "fm_only"
    args = ws_server.resolve_decoder_args(parser.parse_args(base))
    assert args.beta_stop_mode == "manual"
    with pytest.raises(ValueError, match="requires an fm_stop"):
        ws_server.resolve_decoder_args(parser.parse_args(base + ["--beta_stop_mode", "learned"]))
