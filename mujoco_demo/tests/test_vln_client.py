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
