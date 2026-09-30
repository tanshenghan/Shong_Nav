"""Long navigation instructions must remain complete and readable."""

import pytest

from data_pipeline.vlnverse.replay.video_layout import wrap_instruction


def test_complete_instruction_is_wrapped_without_truncation():
    text = "Walk forward into the main room, passing a dining table on your right. Stop between the two open doorways."
    lines = wrap_instruction(text, 31, len)
    assert " ".join(lines) == text
    assert all(len(line) <= 31 for line in lines)
    assert len(lines) > 1


def test_long_unbroken_token_is_split_without_losing_characters():
    text = "abcdefghijklmnopqrstuvwxyz"
    lines = wrap_instruction(text, 8, len)
    assert "".join(lines) == text
    assert all(len(line) <= 8 for line in lines)


@pytest.mark.parametrize("text", [None, "", "   "])
def test_missing_instruction_is_not_silently_replaced(text):
    with pytest.raises(ValueError, match="nonempty instruction"):
        wrap_instruction(text, 100, len)
