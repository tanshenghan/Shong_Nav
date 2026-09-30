#!/usr/bin/env python3
"""Count generated navigation tokens, independently of decoded/executed actions.

Step JSONL rows require scene_id, habitat_episode_id, step and raw_text. Optional
output_token_ids provide exact generated IDs; otherwise only angle-bracket tokens
can be counted from raw_text. Explicit inference_performed=false rows account for
environment steps without a model call and never contribute tokens. The inference
field is accepted as a legacy alias. JSON arrays are also read.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

RVQ = re.compile(r"<act_l(\d+)_(\d+)>")
NAV = re.compile(r"<(?:act_l\d+_\d+|[ao]pos_\d+)>")
SPECIAL = re.compile(r"<[^<>]+>")


def read_rows(path):
    path = Path(path)
    with path.open() as stream:
        first = stream.read(1)
        stream.seek(0)
        if first == "[" or path.suffix == ".json":
            value = json.load(stream)
            if isinstance(value, dict):
                value = value["episodes"]
            if not isinstance(value, list):
                raise ValueError(f"{path}: expected a JSON array or episodes array")
            yield from value
        else:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{number}: invalid JSON") from exc


def expand_paths(patterns):
    paths = set()
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches:
            raise ValueError(f"No files match {pattern}")
        for value in matches:
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"Expected a file: {path}")
            paths.add(path.resolve())
    return sorted(paths)


def load_tokenizer(directory):
    directory = Path(directory)
    token_to_id = {}
    id_to_token = {}
    specials = set()

    def add(token, token_id, special=False):
        token_id = int(token_id)
        if token in token_to_id and token_to_id[token] != token_id:
            raise ValueError(f"Conflicting tokenizer IDs for {token}")
        if token_id in id_to_token and id_to_token[token_id] != token:
            raise ValueError(f"Conflicting tokenizer tokens for ID {token_id}")
        token_to_id[token] = token_id
        id_to_token[token_id] = token
        if special or NAV.fullmatch(token):
            specials.add(token)

    path = directory / "tokenizer.json"
    if path.exists():
        data = json.loads(path.read_text())
        vocab = data.get("model", {}).get("vocab", {})
        if isinstance(vocab, dict):
            for token, token_id in vocab.items():
                add(token, token_id)
        elif isinstance(vocab, list):
            for token_id, item in enumerate(vocab):
                add(item[0], token_id)
        for item in data.get("added_tokens", []):
            add(item["content"], item["id"], item.get("special", False))
    path = directory / "tokenizer_config.json"
    if path.exists():
        data = json.loads(path.read_text())
        for token_id, item in data.get("added_tokens_decoder", {}).items():
            add(item["content"], token_id, item.get("special", False))
    if not token_to_id:
        raise ValueError(f"No tokenizer vocabulary found in {directory}")
    for level in range(3):
        for code in range(256):
            if f"<act_l{level}_{code}>" not in token_to_id:
                raise ValueError(f"Missing RVQ vocabulary token <act_l{level}_{code}>")
    return token_to_id, id_to_token, specials


def episode_key(row):
    return str(row["scene_id"]), str(row["habitat_episode_id"]), int(row.get("rollout_idx", 0))


def index_rows(path):
    result = {}
    if path is not None:
        for row in read_rows(path):
            key = episode_key(row)
            if key in result:
                raise ValueError(f"Duplicate episode in {path}: {key}")
            result[key] = row
    return result


def fraction(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(step_paths, tokenizer_dir, output_dir, results_path=None, expected_path=None):
    token_to_id, id_to_token, specials = load_tokenizer(tokenizer_dir)
    paths = expand_paths(step_paths)
    results = index_rows(results_path)
    expected = index_rows(expected_path)
    frequencies = Counter()
    valid_frequencies = Counter()
    combinations = Counter()
    invalid_reasons = Counter()
    unknown_tokens = Counter()
    sources = Counter()
    counts = Counter()
    episodes = defaultdict(
        lambda: {
            "steps": set(),
            "inferences": 0,
            "non_inferences": 0,
            "valid_rvq": 0,
            "malformed_rvq": 0,
        }
    )

    for path in paths:
        for row in read_rows(path):
            key = episode_key(row)
            step = row["step"]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"{path}: step must be a nonnegative integer")
            state = episodes[key]
            if step in state["steps"]:
                raise ValueError(f"Duplicate inference/environment step: {key}, step={step}")
            state["steps"].add(step)
            counts["logged_environment_steps"] += 1
            inference_performed = row.get("inference_performed", row.get("inference", True))
            if not isinstance(inference_performed, bool):
                raise ValueError(f"{path}: inference_performed must be a boolean")
            if not inference_performed:
                state["non_inferences"] += 1
                counts["explicit_non_inference_steps"] += 1
                continue
            state["inferences"] += 1
            counts["inference_calls"] += 1
            counts["forced_stop_inference_calls"] += bool(row.get("forced_stop", False))
            counts["stop_inference_calls"] += bool(row.get("stop", False))
            counts["reported_decode_failures"] += bool(row.get("decode_failed", False))
            raw = row["raw_text"]
            if not isinstance(raw, str):
                raise ValueError(f"{path}: raw_text must be a string")
            raw_tokens = SPECIAL.findall(raw)
            output_ids = row.get("output_token_ids")
            if output_ids is not None:
                if not isinstance(output_ids, list) or any(
                    isinstance(value, bool) or not isinstance(value, int) for value in output_ids
                ):
                    raise ValueError(f"{path}: output_token_ids must be an integer array")
                tokens = [
                    id_to_token.get(value, f"[unknown_token_id:{value}]") for value in output_ids
                ]
                sources["output_token_ids"] += 1
                if [token for token in tokens if NAV.fullmatch(token)] != [
                    token for token in raw_tokens if NAV.fullmatch(token)
                ]:
                    counts["navigation_text_id_mismatch_calls"] += 1
            else:
                tokens = raw_tokens
                sources["raw_text_special_tokens"] += 1
            frequencies.update(tokens)
            unknown_tokens.update(token for token in tokens if token not in token_to_id)
            parsed = [RVQ.fullmatch(token) for token in tokens if token.startswith("<act_")]
            codes = defaultdict(list)
            reasons = set()
            for match in parsed:
                if match is None:
                    reasons.add("invalid_action_token_syntax")
                    continue
                level, code = map(int, match.groups())
                if level not in range(3) or code not in range(256):
                    reasons.add("out_of_range_action_token")
                else:
                    codes[level].append(code)
            for level in range(3):
                if not codes[level]:
                    reasons.add(f"missing_level_{level}")
                elif len(codes[level]) != 1:
                    reasons.add(f"duplicate_level_{level}")
            if reasons:
                counts["malformed_rvq_calls"] += 1
                state["malformed_rvq"] += 1
                invalid_reasons.update(reasons)
            else:
                counts["valid_rvq_calls"] += 1
                state["valid_rvq"] += 1
                triple = tuple(codes[level][0] for level in range(3))
                combinations[triple] += 1
                valid_frequencies.update(
                    f"<act_l{level}_{code}>" for level, code in enumerate(triple)
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_keys = set(episodes) | set(results) | set(expected)
    coverage_rows = []
    for key in sorted(all_keys):
        state = episodes[key]
        result_steps = results[key].get("steps") if key in results else None
        step_count = len(state["steps"])
        sequence_complete = result_steps is not None and state["steps"] == set(range(result_steps))
        coverage_rows.append(
            {
                "scene_id": key[0],
                "habitat_episode_id": key[1],
                "rollout_idx": key[2],
                "expected": key in expected,
                "has_result": key in results,
                "result_steps": result_steps,
                "logged_steps": step_count,
                "inference_calls": state["inferences"],
                "explicit_non_inference_steps": state["non_inferences"],
                "unlogged_steps": result_steps - step_count if result_steps is not None else None,
                "step_sequence_complete": sequence_complete,
                "valid_rvq_calls": state["valid_rvq"],
                "malformed_rvq_calls": state["malformed_rvq"],
                "success": results.get(key, {}).get("success"),
            }
        )
    logged_keys = {key for key, state in episodes.items() if state["steps"]}
    complete = (
        expected_path is not None
        and results_path is not None
        and bool(expected)
        and set(expected) == set(results) == logged_keys
        and all(row["step_sequence_complete"] for row in coverage_rows)
    )
    token_fields = ["token", "token_id", "count", "count_in_valid_rvq_calls", "per_inference_call"]

    def token_row(token):
        return {
            "token": token,
            "token_id": token_to_id.get(token),
            "count": frequencies[token],
            "count_in_valid_rvq_calls": valid_frequencies[token],
            "per_inference_call": fraction(frequencies[token], counts["inference_calls"]),
        }

    write_csv(
        output / "rvq_tokens.csv",
        ["level", "code", *token_fields],
        (
            {"level": level, "code": code, **token_row(f"<act_l{level}_{code}>")}
            for level in range(3)
            for code in range(256)
        ),
    )
    navigation = {token for token in token_to_id if NAV.fullmatch(token)}
    write_csv(
        output / "navigation_tokens.csv",
        token_fields,
        (token_row(token) for token in sorted(navigation, key=token_to_id.get)),
    )
    write_csv(
        output / "special_tokens.csv",
        token_fields,
        (token_row(token) for token in sorted(specials, key=token_to_id.get)),
    )
    write_csv(
        output / "generated_tokens.csv",
        token_fields,
        (token_row(token) for token, _ in frequencies.most_common()),
    )
    write_csv(
        output / "rvq_combinations.csv",
        ["l0", "l1", "l2", "count", "fraction_valid_calls", "fraction_all_calls"],
        (
            {
                "l0": triple[0],
                "l1": triple[1],
                "l2": triple[2],
                "count": count,
                "fraction_valid_calls": fraction(count, counts["valid_rvq_calls"]),
                "fraction_all_calls": fraction(count, counts["inference_calls"]),
            }
            for triple, count in combinations.most_common()
        ),
    )
    episode_fields = list(coverage_rows[0]) if coverage_rows else ["scene_id", "habitat_episode_id"]
    write_csv(output / "episodes.csv", episode_fields, coverage_rows)
    layers = []
    for level in range(3):
        values = [frequencies[f"<act_l{level}_{code}>"] for code in range(256)]
        total = sum(values)
        probabilities = [value / total for value in values if value] if total else []
        entropy = -sum(probability * math.log2(probability) for probability in probabilities)
        layers.append(
            {
                "level": level,
                "vocabulary_size": 256,
                "used_codes": len(probabilities),
                "usage_fraction": len(probabilities) / 256,
                "emitted_tokens": total,
                "entropy_bits": entropy,
                "effective_codes": 2**entropy if total else 0,
                "top_10_fraction": fraction(sum(sorted(values, reverse=True)[:10]), total),
            }
        )
    summary = {
        "schema_version": 1,
        "status": "complete" if complete else "partial",
        "step_files": [str(path) for path in paths],
        "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
        "counts": {
            name: counts[name]
            for name in (
                "logged_environment_steps",
                "explicit_non_inference_steps",
                "inference_calls",
                "valid_rvq_calls",
                "malformed_rvq_calls",
                "forced_stop_inference_calls",
                "stop_inference_calls",
                "reported_decode_failures",
                "navigation_text_id_mismatch_calls",
            )
        },
        "token_sources": dict(sources),
        "invalid_rvq_reasons": dict(invalid_reasons),
        "unknown_tokens": dict(unknown_tokens),
        "unknown_token_occurrences": sum(unknown_tokens.values()),
        "rvq_layers": layers,
        "observed_valid_combinations": len(combinations),
        "possible_rvq_combinations": 256**3,
        "coverage": {
            "expected_episodes": len(expected),
            "result_episodes": len(results),
            "logged_episodes": len(logged_keys),
            "missing_expected_episodes": len(set(expected) - logged_keys),
            "unexpected_logged_episodes": len(logged_keys - set(expected))
            if expected_path
            else None,
            "result_step_mismatches": sum(
                not row["step_sequence_complete"] for row in coverage_rows if row["has_result"]
            ),
        },
        "notes": [
            "Counts describe generated output tokens; explicit non-inference steps contribute no tokens.",
            "Malformed RVQ outputs contribute token counts but not valid triple counts.",
            "Without output_token_ids, raw_text counts only visible angle-bracket tokens; ordinary text and omitted EOS IDs are unavailable.",
            "Empty raw_text alone does not imply that no model inference occurred.",
            "Complete requires expected/result/logged episode equality and every result environment step accounted for.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        action="append",
        required=True,
        help="JSONL/JSON file paths or quoted globs",
    )
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results", help="Merged evaluation results JSONL")
    parser.add_argument("--expected", help="Expected episode JSON array or episodes manifest")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    summary = analyze(
        [path for group in args.steps for path in group],
        args.tokenizer_dir,
        args.output_dir,
        args.results,
        args.expected,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                **summary["counts"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            indent=2,
        )
    )
    return 2 if args.require_complete and summary["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
