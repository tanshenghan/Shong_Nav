"""Habitat environment server for VLN-CE (R2R / RxR) and ObjectNav (HM3D v1 / MP3D v1 / HM3D-OVON) evaluation.

The package runs inside a habitat-sim / habitat-lab conda environment (Python 3.9)
and exposes one environment instance over ZeroMQ so that a model process running
elsewhere can drive it. It does not depend on torch or on the inference package.
"""

__version__ = "0.1.0"
