"""VLN-CE registry extensions for habitat-lab (dataset ``VLN-CE-v1`` and the VLN-CE measures).

Importing this package registers everything with habitat's global registry; do so before
constructing ``habitat.Env``.
"""

from . import vlnce_dataset, vlnce_measures  # noqa: F401
