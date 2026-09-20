"""Discrete navigation action ids shared by the Habitat environment wrappers."""

STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

# Action id -> display name (used for human-readable termination details).
ACTION_NAMES = {
    STOP: "STOP",
    MOVE_FORWARD: "FORWARD",
    TURN_LEFT: "LEFT",
    TURN_RIGHT: "RIGHT",
}
