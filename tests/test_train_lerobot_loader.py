"""Training reader contracts with embedded PNGs and multi-episode Parquet files."""

from __future__ import annotations

import copy
import io
import json
import pickle
import re
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from lightnav.dataloader.Train_Lerobot_DataLoader import (  # noqa: E402
    FM_TARGET_COLUMNS,
    RGB_KEY,
    TARGET_COLUMNS,
    BetaLerobotCollator,
    LeRobotFrameDataset,
    LightNavTrainingSpec,
    Train_Lerobot_DataLoader,
)
from lightnav.inference.beta_model import BetaTrainBatch  # noqa: E402
from lightnav.inference.frame_preprocessing import rgb_frame_to_model_tensor  # noqa: E402
from lightnav.inference.samples import build_vln_traj_sample  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def files(tmp_path):
    root, model = tmp_path / "data", tmp_path / "model"
    bundle = model / "action_tokenizer"
    bundle.mkdir(parents=True)
    manifest = dict(levels=[256, 256, 256], horizon=10, representation="se2_diff",
                    codebook_files=[f"codebook_l{i}.npy" for i in range(3)],
                    jacobian_weights_file="weights.npy", alpha_file="alpha.json",
                    stop={"l0": 6, "codes": [6, 122, 174]})
    write_json(bundle / "manifest.json", manifest)
    for name in manifest["codebook_files"]:
        np.save(bundle / name, np.zeros((256, 30), np.float32))
    np.save(bundle / "weights.npy", np.ones(30, np.float32))
    write_json(bundle / "alpha.json", {})
    shutil.copytree(bundle, root / "assets/action_tokenizer")
    tiers = [
        dict(name="current", age_lo=0, age_hi=1, mode="dense", pool_spatial=1),
        dict(name="fast", age_lo=2, age_hi=33, mode="dense", pool_spatial=2),
        dict(name="mid", age_lo=34, age_hi=89, mode="burst", pair_stride=6, pool_spatial=2),
        dict(name="long", age_lo=90, age_hi=1000000, mode="span", num_pairs=14, pool_spatial=4),
        dict(name="anchor", age_lo=0, age_hi=0, mode="anchor", num_frames=2, pool_spatial=4),
    ]
    write_json(model / "config.json", {})
    write_json(model / "eval_config.json", dict(version=1,
        common=dict(video_size=[32, 32], pool_enable=True, pool_spatial=2, pool_mode="avg",
                    pool_stage="post_vit", max_seq_len=8192),
        tasks={"vlnce": dict(slowfast_tiers=tiers, predict_horizon=10, video_fps=4, num_history_frames=64,
                             timestamp_relative=True,
                             action_tokenizer={"method": "rvq", "bundle_path": "action_tokenizer"})}))
    rows, episodes = [], []
    offset = 0
    for ep, n in enumerate([140, 5, 1]):
        episodes.append({"episode_index": ep, "tasks": [f"Instruction {ep}: assistant is a word here."],
                         "length": n, "data/chunk_index": 0, "data/file_index": 0,
                         "dataset_from_index": offset, "dataset_to_index": offset + n})
        for t in range(n):
            output = io.BytesIO()
            Image.fromarray(np.full((8, 8, 3), ep * 100 + t, np.uint8)).save(output, format="PNG")
            valid = t + np.arange(1, 31) < n
            clean = np.zeros((30, 3), np.float32)
            clean[valid, 0] = .25
            stop = t == n - 1
            rvq_valid = bool(valid[:10].all() or stop)
            codes = [6, 122, 174] if stop else ([32, 62, 122] if rvq_valid else [-1] * 3)
            rows.append({
                "index": offset + t, "episode_index": ep, "frame_index": t, "task_index": ep,
                "source.episode_id": f"source-{ep}", "source.scene_id": f"scene-{ep}",
                RGB_KEY: {"bytes": output.getvalue(), "path": None},
                "action.apos.token_id": 0 if stop else 100,
                "action.apos.token": "<apos_0>" if stop else "<apos_100>",
                "action.apos.valid_mask": not stop,
                "action.opos.token_id": 0, "action.opos.token": "<opos_0>",
                "action.opos.valid_mask": t != 1,
                "action.rvq.codes": codes,
                "action.rvq.tokens": "".join(f"<act_l{i}_{c}>" for i, c in enumerate(codes)) if rvq_valid else "",
                "action.rvq.valid_mask": rvq_valid,
                "action.fm.clean_se2_diff": clean.tolist(), "action.fm.valid_mask": valid.tolist(),
                "action.fm.future_frame_ids": np.where(valid, t + np.arange(1, 31), -1).tolist(),
                "action.explicit_stop": stop, "action.stop_loss_valid": stop,
            })
        offset += n
    features = {k: {"dtype": "unused", "shape": [1]} for k in TARGET_COLUMNS}
    features.update({RGB_KEY: {"dtype": "image", "shape": [8, 8, 3]},
                     "action.fm.clean_se2_diff": {"dtype": "float32", "shape": [30, 3]},
                     "action.fm.valid_mask": {"dtype": "bool", "shape": [30]}})
    write_json(root / "meta/info.json", dict(codebase_version="v3.0", fps=1, features=features,
        total_episodes=len(episodes), total_frames=offset,
        data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"))
    data_path, meta_path = root / "data/chunk-000/file-000.parquet", root / "meta/episodes/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True)
    meta_path.parent.mkdir(parents=True)
    # Row groups intentionally do not coincide with episode boundaries.
    pq.write_table(pa.Table.from_pylist(rows), data_path, row_group_size=17)
    pq.write_table(pa.Table.from_pylist(episodes), meta_path)
    return root, model, rows


@pytest.fixture
def dataset(files):
    root, model, _ = files
    return LeRobotFrameDataset(root, LightNavTrainingSpec.from_checkpoint(model), cache_frames=2)


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.tokens = {"<|im_end|>": 4}

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant\n":
            return [1, 2, 3]
        if text not in self.tokens:
            self.tokens[text] = 100 + len(self.tokens)
        return [self.tokens[text]]

    def convert_ids_to_tokens(self, index):
        return next(k for k, v in self.tokens.items() if v == index)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def process_sample(self, sample):
        segments = sample["video_segments"]
        prefix = [17] * (sum(len(s["frame_indices"]) for s in segments) + 2)
        text = sample["conversations"][-1]["value"]
        answer = [self.tokenizer.encode(t)[0] for t in re.findall(r"<[^>]+>", text)]
        ids = torch.tensor(prefix + [1, 2, 3] + answer + [4, 3])
        original = torch.tensor([[len(s["frame_indices"]) // 2, 8, 8] for s in segments])
        grid = original.clone()
        for i, s in enumerate(segments):
            grid[i, 1:] //= s["pool_spatial"]
        return dict(input_ids=ids, labels=ids.clone(), position_ids=torch.zeros(3, len(ids)),
                    pixel_values_videos=torch.full((int(original.prod(dim=1).sum()), 24), float(len(prefix))),
                    video_grid_thw=grid, _original_video_grid_thw=original,
                    _post_vit_pool_factors=[s["pool_spatial"] for s in segments])


class FakeCollator(BetaLerobotCollator):
    def _get_processor(self):
        if self._processor is None:
            self._processor = FakeProcessor()
        return self._processor


def test_fm_only_needs_no_rvq_assets_or_token_columns(files):
    root, model, _ = files
    shutil.rmtree(root / "assets/action_tokenizer")
    shutil.rmtree(model / "action_tokenizer")
    info = json.loads((root / "meta/info.json").read_text())
    info["features"] = {k: v for k, v in info["features"].items() if k in FM_TARGET_COLUMNS + [RGB_KEY]}
    write_json(root / "meta/info.json", info)
    path = root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(path, columns=FM_TARGET_COLUMNS + [RGB_KEY])
    pq.write_table(table, path, row_group_size=17)
    spec = LightNavTrainingSpec.from_checkpoint(model, training_mode="fm_only")
    assert not spec.rvq_path and "action_tokenizer" not in spec.params
    ds = LeRobotFrameDataset(root, spec)
    items = [ds[0], ds[139]]
    assert ds.rvq_manifest is None
    assert all(not item["tokens"] and not item["token_valid"] for item in items)
    assert "trajectory tokens" not in items[0]["sample"]["conversations"][0]["value"]
    batch = FakeCollator(spec)(items)
    assert (batch.labels == -100).all()
    assert (batch.answer_start == batch.labels.shape[1]).all()
    assert not (batch.model_inputs["input_ids"] == 4).any()  # no assistant EOS/answer
    assert batch.fm_valid_mask[0].all() and not batch.fm_valid_mask[1].any()


@pytest.mark.parametrize("mode", ["fm_only", "ce_fm"])
def test_ten_step_targets_are_consecutive_prefix_with_episode_tail_masks(files, mode):
    root, model, rows = files
    spec = LightNavTrainingSpec.from_checkpoint(model, training_mode=mode)
    ds = LeRobotFrameDataset(root, spec, fm_horizon=10)
    assert len(ds) == len(rows)  # shortening a target must not drop training frames
    # Distinct values detect downsampling/interpolation or zero-padding mistakes.
    row = copy.deepcopy(rows[0])
    row["action.fm.clean_se2_diff"] = np.arange(90, dtype=np.float32).reshape(30, 3).tolist()
    _, _, clean, valid = ds._targets(ds.episodes[0], 0, row)
    torch.testing.assert_close(clean, torch.arange(30, dtype=torch.float32).reshape(10, 3))
    assert valid.shape == (10,) and valid.all()
    items = [ds[0], ds[134], ds[139], ds[145]]
    batch = FakeCollator(spec)(items)
    assert batch.fm_clean_se2_diff.shape == (4, 10, 3)
    assert batch.fm_valid_mask.shape == (4, 10)
    assert batch.fm_valid_mask.sum(dim=1).tolist() == [10, 5, 0, 0]
    assert all(item["info"]["fm_horizon"] == 10 for item in items)
    if mode == "fm_only":
        assert (batch.labels == -100).all()


@pytest.mark.parametrize("horizon", [0, 31, 1.5, True])
def test_reader_rejects_invalid_fm_horizon(files, horizon):
    root, model, _ = files
    with pytest.raises(ValueError, match="fm_horizon"):
        Train_Lerobot_DataLoader(root, model, fm_horizon=horizon)


def test_loader_passes_ten_step_horizon_to_dataset(files):
    root, model, _ = files
    loader = Train_Lerobot_DataLoader(root, model, fm_horizon=10, training_mode="fm_only")
    assert loader.dataset[0]["fm_clean"].shape == (10, 3)


def test_metadata_index_is_complete_and_subset_uses_original_episode_ids(files):
    root, model, _ = files
    ds = LeRobotFrameDataset(root, LightNavTrainingSpec.from_checkpoint(model), episode_indices=[2, 1])
    assert len(ds) == 6
    assert ds.locate(0)[0].index == 2
    assert ds.locate(1)[0].index == 1
    assert ds[1]["info"]["global_frame_index"] == 140
    with pytest.raises(IndexError):
        ds[-1]
    with pytest.raises(ValueError, match="unique"):
        LeRobotFrameDataset(root, ds.spec, episode_indices=[1, 1])


@pytest.mark.parametrize("t", [0, 1, 20, 35, 89, 120, 139])
def test_history_and_prompt_match_existing_inference_bit_for_bit(dataset, t):
    item = dataset[t]
    frames = torch.stack([rgb_frame_to_model_tensor(np.full((8, 8, 3), f, np.uint8), (32, 32)) for f in range(t + 1)])
    bundle = SimpleNamespace(slowfast_tiers=dataset.spec.params["slowfast_tiers"], action_method="rvq",
                             predict_horizon=10, video_fps=4)
    reference = build_vln_traj_sample(frames, dataset.episodes[0].instruction, None, bundle)
    assert item["sample"]["conversations"][0] == reference["conversations"][0]
    for actual, expected in zip(item["sample"]["video_segments"], reference["video_segments"], strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            if key == "video":
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            else:
                assert actual[key] == expected[key]


def test_only_past_rgb_is_decoded_and_never_from_another_episode(dataset, monkeypatch):
    original, seen = dataset._image, []

    def record(ep, frame):
        seen.append((ep.index, frame))
        return original(ep, frame)

    monkeypatch.setattr(dataset, "_image", record)
    item = dataset[142]  # episode 1, t=2
    assert all(ep == 1 and 0 <= t <= 2 for ep, t in seen)
    assert max(t for _, t in seen) == 2
    assert item["info"]["instruction"] == dataset.episodes[1].instruction
    assert all(s["total_frames"] == 3 for s in item["sample"]["video_segments"])
    assert len(dataset._frames) <= 2
    assert len(dataset._tables) <= dataset.cache_row_groups


def test_fm_is_the_existing_chunk_and_partial_rvq_is_masked(dataset, files):
    item = dataset[132]
    assert item["fm_clean"].shape == (30, 3)
    assert item["fm_valid"].sum() == 7
    assert item["token_valid"] == [True, True, False, False, False]
    assert item["tokens"][-3:] == ["<act_l0_0>", "<act_l1_0>", "<act_l2_0>"]
    np.testing.assert_array_equal(item["fm_clean"].numpy(), np.asarray(files[2][132]["action.fm.clean_se2_diff"], np.float32))
    assert files[2][132]["action.rvq.codes"] == [-1, -1, -1]


def test_terminal_and_unknown_pointing_masks_remain_independent(dataset):
    stop = dataset[139]
    assert stop["tokens"] == ["<apos_0>", "<opos_0>", "<act_l0_6>", "<act_l1_122>", "<act_l2_174>"]
    assert stop["token_valid"] == [False, True, True, True, True]
    assert stop["fm_valid"].sum() == 0
    assert dataset[0]["tokens"][1] == dataset[1]["tokens"][1] == "<opos_0>"
    assert dataset[0]["token_valid"][1] and not dataset[1]["token_valid"][1]


def test_collator_pads_boundaries_and_concatenates_visual_inputs(dataset):
    items = [dataset[20], dataset[139], dataset[132]]
    collator = FakeCollator(dataset.spec)
    processed = [collator._process(item) for item in items]
    batch = collator(items)
    assert isinstance(batch, BetaTrainBatch)
    assert batch.fm_clean_se2_diff.shape == (3, 30, 3)
    assert batch.fm_valid_mask.sum(dim=1).tolist() == [30, 0, 7]
    assert (batch.labels != -100).sum(dim=1).tolist() == [6, 5, 3]
    for i, (result, _, start) in enumerate(processed):
        shift = batch.labels.shape[1] - len(result["input_ids"])
        assert batch.answer_start[i] == shift + start
        assert (batch.labels[i, :shift + start] == -100).all()
        assert batch.model_inputs["attention_mask"][i, :shift].sum() == 0
    start = batch.answer_start[1].item()
    assert batch.labels[1, start] == -100  # ignored first answer token must NOT move boundary
    assert batch.model_inputs["input_ids"][1, start] == collator._processor.tokenizer.encode("<apos_0>")[0]
    for key in ["pixel_values_videos", "video_grid_thw", "_original_video_grid_thw"]:
        torch.testing.assert_close(batch.model_inputs[key], torch.cat([r[key] for r, _, _ in processed]))
    assert "position_ids" not in batch.model_inputs and "labels" not in batch.model_inputs
    assert all("world" not in key and "depth" not in key for key in batch.model_inputs)


def test_truncated_answer_or_excessive_length_fails_instead_of_silent_training(dataset):
    item = dataset[0]
    collator = FakeCollator(dataset.spec)
    processor = collator._get_processor()
    original = processor.process_sample

    def truncated(sample):
        result = original(sample)
        result["input_ids"] = result["input_ids"][:-4]
        return result

    processor.process_sample = truncated
    with pytest.raises(ValueError, match="truncated"):
        collator([item])
    processor.process_sample = original
    collator.spec.params["max_seq_len"] = 2
    with pytest.raises(ValueError, match="refusing to truncate"):
        collator([item])


@pytest.mark.parametrize("field,value", [
    ("action.fm.valid_mask", [False] * 30),
    ("action.fm.future_frame_ids", [-1] * 30),
    ("action.rvq.codes", [6, 62, 122]),
    ("action.apos.token", "<apos_999>"),
    ("action.explicit_stop", True),
])
def test_inconsistent_supervision_is_rejected(dataset, files, field, value):
    row = copy.deepcopy(files[2][0])
    row[field] = value
    with pytest.raises(ValueError):
        dataset._targets(dataset.episodes[0], 0, row)


def test_changed_codebook_rejected_before_loading_frames(files):
    root, model, _ = files
    (model / "action_tokenizer/weights.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset mismatch"):
        LeRobotFrameDataset(root, LightNavTrainingSpec.from_checkpoint(model))


def test_pickling_drops_open_parquet_and_processor_objects(dataset):
    dataset[0]
    assert dataset._files and dataset._frames
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._files and not restored._frames and not restored._tables
    assert restored[0]["info"] == dataset[0]["info"]
    collator = FakeCollator(dataset.spec)
    collator([dataset[0]])
    assert pickle.loads(pickle.dumps(collator))._processor is None


def test_loader_yields_beta_batch_and_epoch_order_is_reproducible(files):
    root, model, _ = files
    loader = Train_Lerobot_DataLoader(root, model, batch_size=2, seed=19)
    loader.collate_fn = FakeCollator(loader.spec)
    assert isinstance(next(iter(loader)), BetaTrainBatch)
    loader.set_epoch(4)
    first = list(loader.sampler)
    loader.set_epoch(4)
    assert list(loader.sampler) == first
    loader.set_epoch(5)
    assert list(loader.sampler) != first


def test_distributed_sampler_shards_same_dataset_without_starting_process_group(files):
    root, model, _ = files
    a = Train_Lerobot_DataLoader(root, model, rank=0, world_size=2, shuffle=False)
    b = Train_Lerobot_DataLoader(root, model, rank=1, world_size=2, shuffle=False)
    left, right = set(a.sampler), set(b.sampler)
    assert not left & right
    assert left | right == set(range(146))


def test_spawn_workers_return_the_same_batches_as_main_process(files):
    root, model, _ = files
    serial = Train_Lerobot_DataLoader(root, model, shuffle=False, batch_size=2, episode_indices=[1])
    serial.collate_fn = FakeCollator(serial.spec)
    parallel = Train_Lerobot_DataLoader(root, model, shuffle=False, batch_size=2,
                                       episode_indices=[1], num_workers=2)
    parallel.collate_fn = FakeCollator(parallel.spec)
    try:
        for a, b in zip(serial, parallel, strict=True):
            assert a.sample_info == b.sample_info
            torch.testing.assert_close(a.labels, b.labels)
            torch.testing.assert_close(a.fm_clean_se2_diff, b.fm_clean_se2_diff)
            torch.testing.assert_close(a.model_inputs["pixel_values_videos"], b.model_inputs["pixel_values_videos"])
    finally:
        if parallel._iterator is not None:
            parallel._iterator._shutdown_workers()
