import pytest
from vln_mujoco.vln_client import VlnClient, normalize_server_url


def test_normalize_server_url() -> None:
    assert normalize_server_url("localhost:8000/ws") == "ws://localhost:8000/ws"
    assert normalize_server_url("wss://example.com/vln") == "wss://example.com/vln"


@pytest.mark.parametrize("value", ["http://example.com", "ws://user@example.com", "ws://host:99999"])
def test_reject_invalid_server_url(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_server_url(value)


def test_parse_nested_waypoints() -> None:
    assert VlnClient._parse_waypoints([[0.2, 0.0, 0.1], {"actions": [0.4, 0.1, 0.2]}]) == (
        (0.2, 0.0, 0.1),
        (0.4, 0.1, 0.2),
    )


def test_stop_discards_late_prediction_and_keeps_latest_frame():
    import queue
    import threading

    client = VlnClient.__new__(VlnClient)
    client._lock = threading.Lock()
    client._instruction = 'walk'
    client._connected = True
    client._session = 1
    client._pending_frame = None
    client._results = queue.Queue()
    assert client.offer_frame(1, b'old')
    assert client.offer_frame(2, b'new')
    assert client._take_frame(1) == (2, b'new')
    result = object()
    client._publish_result(1, result)
    client.stop()
    client._publish_result(1, result)
    assert client.take_results() == []
