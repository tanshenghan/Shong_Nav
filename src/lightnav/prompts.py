"""Prompt templates for the navigation and tracking tasks.

Every string here is a trained-in contract: the prompt must match, byte for byte, the
one the checkpoint was trained with. Do not edit wording.

Template variables:
    {task}    -- natural-language task instruction
    {horizon} -- number of actions to predict (VLN_PROMPT_TEMPLATE* only)
    {videos}  -- one ``<video>`` placeholder per video segment (UNIFIED_TRAJ_PROMPT_TEMPLATE)

Each ``<video>`` placeholder is consumed by the Qwen3-VL processor and maps to one
entry of ``sample["video_segments"]``.
"""

# -- VLN-CE (discrete actions: <move_forward>, <turn_left>, <turn_right>, <stop>) --
# Kept for reference: the discrete-action family is not served by this package.

VLN_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a sequence of observations <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be in the beginning, middle or end of the task. "
    "Analyze this video to determine your next {horizon} actions. "
)

VLN_PROMPT_TEMPLATE_POOLED = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given history observations <video> and current observation <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be in the beginning, middle or end of the task. "
    "Analyze these observations to determine your next {horizon} actions. "
)

# -- VLN-CE (single trajectory-id token output) --
# Same task framing as VLN_PROMPT_TEMPLATE but the output contract is one <traj_k>
# token whose id indexes a precomputed centroid of the trajectory vocabulary.

VLN_TRAJ_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a sequence of observations <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be in the beginning, middle or end of the task. "
    "Predict your future trajectory as a single trajectory id token. "
)

VLN_TRAJ_PROMPT_TEMPLATE_POOLED = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given history observations <video> and current observation <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be in the beginning, middle or end of the task. "
    "Predict your future trajectory as a single trajectory id token. "
)

# -- Tracking (single trajectory-id token from a precomputed vocabulary) --
# The output is a single <traj_k> special token whose id indexes a centroid in the
# tracking trajectory vocabulary. K and horizon are baked into the vocabulary when the
# checkpoint is built; the prompt stays vocabulary-agnostic.

TRACKING_PROMPT_TEMPLATE = (
    "You are a mobile robot performing a person-tracking task. "
    "You have been given a sequence of observations <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "Predict your future trajectory as a single trajectory id token. "
)

TRACKING_PROMPT_TEMPLATE_POOLED = (
    "You are a mobile robot performing a person-tracking task. "
    "You have been given history observations <video> and current observation <video>. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "Predict your future trajectory as a single trajectory id token. "
)

# -- Unified trajectory-token prompt (navigation + tracking) --
# Task-agnostic framing; the task specifics (goal navigation / person following) live
# in {task}. Output contract is a trajectory id token (tracking checkpoints emit an
# extra <tpos_k> prefix learned from labels, not described here).
#
# The {videos} slot is filled at build time with one <video> placeholder per
# video_segment (oldest -> newest), so a SINGLE template covers single-segment,
# history/current 2-segment, AND SlowFast N-segment inputs. Each <video> carries its
# own <t.t seconds> timestamp prefix downstream. Selected by eval_config
# ``prompt_style: "unified_traj"`` / SlowFast checkpoints.

UNIFIED_TRAJ_PROMPT_TEMPLATE = (
    "You are a mobile robot. You are given visual observations over time, "
    "ordered from earliest to most recent: {videos}. "
    "Your assigned task is: <navigation_task>{task}</navigation_task>. "
    "You may be at the beginning, middle, or end of the task. "
    "Predict your future trajectory as a trajectory id token. "
)


def build_video_block(num_segments: int) -> str:
    """Space-joined ``<video>`` placeholders, one per video segment."""
    if num_segments < 1:
        raise ValueError(f"num_segments must be >= 1, got {num_segments}")
    return " ".join(["<video>"] * num_segments)


# -- RVQ output-contract rewrite --
# An RVQ action tokenizer emits D coarse->fine <act_l*> tokens instead of one <traj_k>,
# so the trailing "output format" sentence is swapped to match. Only that sentence
# changes -- task framing, the {videos} slot and timestamps are preserved. Applied when
# eval_config ``action_tokenizer.method == "rvq"``. Only the unified template ends with
# the flat sentence below; the per-task templates ("...a single trajectory id token.")
# are intentionally NOT matched, so a miss fails loudly instead of silently keeping the
# flat wording.
_RVQ_OUTPUT_SENTENCE = "Predict your future trajectory as a sequence of coarse-to-fine trajectory tokens. "
_FLAT_OUTPUT_SENTENCE = "Predict your future trajectory as a trajectory id token. "
CONTINUOUS_TRAJ_PROMPT_TEMPLATE = (
    UNIFIED_TRAJ_PROMPT_TEMPLATE.removesuffix(_FLAT_OUTPUT_SENTENCE)
    + "Predict your future trajectory. "
)


def to_rvq_prompt(template: str) -> str:
    """Swap the unified traj template's output sentence for the RVQ one.

    Raises if ``template`` does not end with the expected traj output sentence:
    callers invoke this only for RVQ checkpoints, so a miss is a prompt/method
    mismatch (an RVQ checkpoint without the unified prompt style) that must fail
    loudly rather than silently keep the flat wording.
    """
    if not template.endswith(_FLAT_OUTPUT_SENTENCE):
        raise ValueError(
            f"to_rvq_prompt: template does not end with the expected traj output sentence "
            f"{_FLAT_OUTPUT_SENTENCE!r}; rvq datasets must use prompt_style='unified_traj'. "
            f"template={template!r}"
        )
    return template[: -len(_FLAT_OUTPUT_SENTENCE)] + _RVQ_OUTPUT_SENTENCE
