"""Token usage must preserve generation counts and detect incomplete rollouts."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "analyze_r2r_tokens", Path(__file__).parents[1] / "scripts" / "analyze_r2r_tokens.py"
)
analyzer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyzer)


@pytest.fixture
def tokenizer(tmp_path):
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    tokens = [f"<act_l{level}_{code}>" for level in range(3) for code in range(256)]
    tokens += ["<apos_0>", "<opos_0>", "<|im_end|>"]
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"vocab": {"ordinary": 1}},
                "added_tokens": [
                    {"id": 1000 + index, "content": token, "special": True}
                    for index, token in enumerate(tokens)
                ],
            }
        )
    )
    return directory


def save(path, value):
    path.write_text(json.dumps(value))
    return path


def row(step, raw="<act_l0_0><act_l1_0><act_l2_0>", **kwargs):
    return {
        "scene_id": "scene",
        "habitat_episode_id": "10",
        "step": step,
        "raw_text": raw,
        **kwargs,
    }


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_complete_counts_keep_zero_vocab_and_exclude_non_inference(tmp_path, tokenizer):
    steps = save(
        tmp_path / "steps.json",
        [
            row(0, forced_stop=True),
            row(1, "<act_l0_1><act_l1_2><act_l2_3>"),
            row(2, "", inference_performed=False, inference=True, forced_stop=True),
        ],
    )
    expected = save(tmp_path / "expected.json", [{"scene_id": "scene", "habitat_episode_id": "10"}])
    results = save(
        tmp_path / "results.json", [{"scene_id": "scene", "habitat_episode_id": "10", "steps": 3}]
    )
    output = tmp_path / "report"
    summary = analyzer.analyze([steps], tokenizer, output, results, expected)
    assert summary["status"] == "complete"
    assert summary["counts"]["inference_calls"] == 2
    assert summary["counts"]["explicit_non_inference_steps"] == 1
    assert summary["counts"]["forced_stop_inference_calls"] == 1
    tokens = csv_rows(output / "rvq_tokens.csv")
    assert len(tokens) == 768
    assert tokens[0]["token_id"] == "1000"
    assert tokens[255]["count"] == "0"
    assert summary["rvq_layers"][0]["used_codes"] == 2
    assert len(csv_rows(output / "rvq_combinations.csv")) == 2


def test_malformed_unknown_and_empty_outputs_remain_in_denominator(tmp_path, tokenizer):
    steps = save(
        tmp_path / "steps.json",
        [
            row(0),
            row(1, "<act_l0_0><act_l0_1><act_l1_0><act_l2_0>"),
            row(2, "<act_l0_999><act_l1_0><act_l2_0><mystery>"),
            row(3, ""),
        ],
    )
    summary = analyzer.analyze([steps], tokenizer, tmp_path / "report")
    assert summary["counts"]["inference_calls"] == 4
    assert summary["counts"]["valid_rvq_calls"] == 1
    assert summary["counts"]["malformed_rvq_calls"] == 3
    assert summary["unknown_token_occurrences"] == 2
    assert summary["invalid_rvq_reasons"]["duplicate_level_0"] == 1
    assert summary["status"] == "partial"


def test_ids_are_authoritative_and_base_vocabulary_is_supported(tmp_path, tokenizer):
    steps = save(
        tmp_path / "steps.json",
        [
            row(
                0,
                "<act_l0_1><act_l1_0><act_l2_0>",
                output_token_ids=[1000, 1256, 1512, 1, 99999],
            )
        ],
    )
    output = tmp_path / "report"
    summary = analyzer.analyze([steps], tokenizer, output)
    assert summary["counts"]["navigation_text_id_mismatch_calls"] == 1
    assert summary["counts"]["valid_rvq_calls"] == 1
    assert summary["unknown_tokens"] == {"[unknown_token_id:99999]": 1}
    generated = {item["token"]: item for item in csv_rows(output / "generated_tokens.csv")}
    assert generated["ordinary"]["token_id"] == "1"
    assert generated["<act_l0_0>"]["count"] == "1"
    assert "<act_l0_1>" not in generated


def test_duplicate_steps_fail_even_across_files(tmp_path, tokenizer):
    first = save(tmp_path / "first.json", [row(0)])
    second = save(tmp_path / "second.json", [row(0)])
    with pytest.raises(ValueError, match="Duplicate inference/environment step"):
        analyzer.analyze([first, second], tokenizer, tmp_path / "report")


def test_missing_steps_and_episodes_prevent_complete_status(tmp_path, tokenizer):
    steps = save(tmp_path / "steps.json", [row(1)])
    expected = save(
        tmp_path / "expected.json",
        [
            {"scene_id": "scene", "habitat_episode_id": "10"},
            {"scene_id": "scene", "habitat_episode_id": "11"},
        ],
    )
    results = save(
        tmp_path / "results.json",
        [
            {"scene_id": "scene", "habitat_episode_id": "10", "steps": 2},
        ],
    )
    output = tmp_path / "report"
    code = analyzer.main(
        [
            "--steps",
            str(steps),
            "--tokenizer-dir",
            str(tokenizer),
            "--output-dir",
            str(output),
            "--expected",
            str(expected),
            "--results",
            str(results),
            "--require-complete",
        ]
    )
    assert code == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["coverage"]["missing_expected_episodes"] == 1
    assert summary["coverage"]["result_step_mismatches"] == 1
    assert csv_rows(output / "episodes.csv")[0]["unlogged_steps"] == "1"


def test_jsonl_and_config_decoder_fallback(tmp_path, tokenizer):
    data = json.loads((tokenizer / "tokenizer.json").read_text())
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {str(item["id"]): item for item in data["added_tokens"]},
            }
        )
    )
    (tokenizer / "tokenizer.json").unlink()
    steps = tmp_path / "steps.jsonl"
    steps.write_text(json.dumps(row(0)) + "\n\n" + json.dumps(row(1)) + "\n")
    summary = analyzer.analyze([str(tmp_path / "*.jsonl")], tokenizer, tmp_path / "report")
    assert summary["counts"]["valid_rvq_calls"] == 2
    assert summary["rvq_layers"][0]["emitted_tokens"] == 2
