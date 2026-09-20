import pytest

from data_pipeline.vlnverse.replay.render_parallel import frame_ranges


@pytest.mark.parametrize(("count", "workers"), [(1, 4), (7, 4), (1001, 4)])
def test_shards_cover_every_pose_once_without_empty_ranges(count, workers):
    ranges = frame_ranges(count, workers)
    assert all(start < stop for start, stop in ranges)
    assert [i for start, stop in ranges for i in range(start, stop)] == list(range(count))


@pytest.mark.parametrize(("count", "workers"), [(0, 4), (7, 0), (7, -1), (1.2, 2)])
def test_invalid_render_partition_fails(count, workers):
    with pytest.raises(ValueError):
        frame_ranges(count, workers)
