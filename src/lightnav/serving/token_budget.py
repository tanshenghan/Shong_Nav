"""Per-step decode token budget derived from the checkpoint's token families."""

from __future__ import annotations


def action_token_count(rvq_bundle=None) -> int:
    """Action tokens a step emits: one ``<traj_k>`` (flat) or one per RVQ level."""
    return len(rvq_bundle.levels) if rvq_bundle is not None else 1


def probe_grounding_tokens(tokenizer) -> int:
    """Grounding-prefix length inferred from the checkpoint tokenizer's token families.

    A dual-pointing checkpoint prepends ``<apos_k><opos_k>`` to the action tokens; a v2
    tracking checkpoint prepends ``<tpos_k>``. Probing the vocabulary beats a flag
    because swapping the served checkpoint is a config change, not a code change, and
    a hand-set count would silently go stale exactly then.

    Vocabulary presence is an UPPER BOUND on what a step emits (a checkpoint trained
    on a mix carries every family but emits one per source), and this returns the sum
    of what is present, so the cap can only ever be loose, never truncating.
    """

    def _has(token: str) -> bool:
        tid = tokenizer.convert_tokens_to_ids(token)
        return tid is not None and tid != getattr(tokenizer, "unk_token_id", None)

    pointing = 2 if _has("<apos_0>") else 0
    tpos = 1 if _has("<tpos_0>") else 0
    return pointing + tpos


def decode_token_budget(grounding_tokens: int, rvq_bundle=None) -> int:
    """Per-step generation cap that stops right after the last action token.

    Capping at exactly the tokens a step emits saves the extra decode step spent
    reaching ``eos``. ``max_tokens`` only truncates, so the greedy prefix is provably
    identical.

    ``grounding_tokens`` is the length of the grounding prefix the CHECKPOINT emits
    before the action tokens; it is not derivable from the served task. Known
    checkpoint layouts:

      | checkpoint family                       | prefix                  | tokens |
      |-----------------------------------------|-------------------------|--------|
      | tracking, legacy                        | ``<tpos_k>``            | 1      |
      | tracking + pointing                     | ``<opos_k>``            | 1      |
      | nav/vln + pointing                      | ``<apos_A><opos_O>``    | 2      |
      | vln, legacy flat                        | (none)                  | 0      |

    A pointing checkpoint also OMITS a channel whose id is -1 (unlabeled), so its
    prefix length varies per step — pass the maximum. Too large costs one decode
    step; too small truncates the last action token, which fails the RVQ level-count
    check as a loud rc=500 (never silently wrong waypoints), but still fails.

    The server derives it from the checkpoint tokenizer (:func:`probe_grounding_tokens`).
    """
    if grounding_tokens < 0:
        raise ValueError(f"grounding_tokens must be >= 0, got {grounding_tokens}")
    return int(grounding_tokens) + action_token_count(rvq_bundle)
