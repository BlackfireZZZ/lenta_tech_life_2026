"""Detection-time inference helpers.

- `sahi_adapter`  : SAHI tiled inference (+3–8 mAP for small objects).
- `tta`           : Test-time augmentation (hflip + multi-scale) + WBF fusion.
- `wbf`           : Weighted Box Fusion across multiple detectors.

Each submodule is imported on demand — this package's __init__ stays lean
so that `wbf` (pure Python, no cv2) is importable without OpenCV.
"""
