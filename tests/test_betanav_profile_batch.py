from types import SimpleNamespace

import pytest

from lightnav.training.profile_batch import select_frame


def test_profile_selects_dataset_local_frame_with_complete_future():
    dataset = SimpleNamespace(episodes=[SimpleNamespace(index=9, length=40),
                                        SimpleNamespace(index=2, length=160)])
    assert select_frame(dataset) == 40 + 129
    assert select_frame(dataset, 20) == 40 + 20
    with pytest.raises(ValueError, match="30 future"):
        select_frame(dataset, 130)
