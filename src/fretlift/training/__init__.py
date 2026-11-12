"""Training your own transcription model (Phase 2).

Install with ``pip install -e ".[train]"``. See docs/TRAINING.md.

Modules that only need NumPy/SciPy (labels, manifests, decoding, augmentation,
features) import without PyTorch, so they can be tested and used for data prep
on any machine. ``model``, ``data``, ``inference`` and ``train`` need PyTorch.
"""
