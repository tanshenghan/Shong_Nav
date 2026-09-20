"""lightnav: standalone inference for trajectory-token vision-language navigation models.

Given a Hugging Face checkpoint plus a trajectory vocabulary (a flat
``centroids_whole_chunk_K{K}_h{H}.npy`` or an RVQ action-tokenizer bundle), turn
(video frames + instruction) into predicted trajectory tokens and decode them into an
``(H, 3)`` waypoint chunk in the robot-local frame ``[forward_m, lateral_m, yaw_rad]``.

Backends: ``hf`` (transformers ``generate``) and ``vllm_local`` (in-process vLLM).
"""

__version__ = "0.1.0"
