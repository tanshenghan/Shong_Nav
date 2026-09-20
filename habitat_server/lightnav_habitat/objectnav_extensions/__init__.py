"""ObjectNav registry extensions for habitat-lab (tolerant ``ObjectNav-v1`` dataset loader).

Importing this package re-registers ``ObjectNav-v1``; do so before constructing ``habitat.Env``.
"""

from . import objectnav_dataset  # noqa: F401
