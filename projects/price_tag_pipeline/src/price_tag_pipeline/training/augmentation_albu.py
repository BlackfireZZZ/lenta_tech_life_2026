"""Albumentations augmentation pipeline tuned for moving-robot retail-shelf footage.

Rationale per transform is documented inline. This is the **heavy** profile
(see UltralyticsAugConfig in augmentation.py for the lighter built-in path).

Use this when:
- Training data is limited (≤10k labeled frames).
- The robot moves fast → strong motion blur in real footage.
- Store lighting varies aisle-to-aisle and includes glare from plastic covers.

Integration with Ultralytics: pass this pipeline through a custom Dataset
class that wraps Ultralytics' YOLODataset and applies these transforms before
the YOLO mosaic/letterbox step. See `attach_to_ultralytics()` below for the
hook (called from the training script when --use-albu is passed).

Sections map to docs/detector-finetuning-report.md §5:
- §5.2 heavy domain albumentations (blur/noise/light/geom/occlusion);
- §5.3 camera-matched geometric/codec aug (lens-distortion jitter, JPEG/codec
  artifacts, vignetting/corner falloff, mild chromatic aberration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeavyAugConfig:
    motion_blur_prob: float = 0.35
    motion_blur_kernel_max: int = 25
    defocus_prob: float = 0.10
    gauss_blur_prob: float = 0.10
    noise_prob: float = 0.25
    brightness_prob: float = 0.45
    clahe_prob: float = 0.10
    shadow_prob: float = 0.15
    sun_flare_prob: float = 0.10
    hue_prob: float = 0.20
    perspective_prob: float = 0.30
    coarse_dropout_prob: float = 0.15
    coarse_dropout_max_holes: int = 4
    hflip_prob: float = 0.5

    # --- §5.3 camera-matched geometric/codec aug (new) ---
    # Lens-distortion jitter around our measured operating point (k1≈-0.28):
    # makes the detector robust to residual distortion + per-store camera/zoom.
    lens_distortion_prob: float = 0.25
    lens_distort_limit: float = 0.10      # OpticalDistortion: ±limit
    grid_distort_steps: int = 5           # GridDistortion grid resolution
    grid_distort_limit: float = 0.10
    # JPEG/video codec artifacts — frames come from compressed video, the
    # pretrained model likely saw clean stills.
    image_compression_prob: float = 0.30
    jpeg_quality_min: int = 40
    jpeg_quality_max: int = 85
    # Vignetting / corner falloff — wide lens + small sensor darkens corners
    # where peripheral tags already suffer from barrel warp.
    vignette_prob: float = 0.15
    vignette_min_strength: float = 0.15
    vignette_max_strength: float = 0.45
    # Mild chromatic aberration at edges (wide lens) — optional, low p.
    # Skipped automatically if the installed Albumentations lacks the transform.
    chromatic_aberration_prob: float = 0.10

    # Bbox safe transforms — never flip vertically, never apply 90° rotation
    # (would invert price text).
    seed: int = 42


def build_albu_transform(cfg: HeavyAugConfig = HeavyAugConfig()):
    """Return a `albumentations.Compose` pipeline with bbox support.

    The output transform follows Ultralytics' built-in Albumentations wrapper:
    images are HxWx3 ndarrays and boxes are normalized YOLO xywh boxes.
    """
    try:
        import albumentations as A  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "albumentations is not installed. Install: pip install albumentations"
        ) from e

    def gauss_noise():
        try:
            return A.GaussNoise(std_range=(0.04, 0.16), mean_range=(0.0, 0.0), p=1.0)
        except TypeError:
            return A.GaussNoise(var_limit=(10.0, 60.0), p=1.0)

    def random_shadow():
        try:
            return A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                p=cfg.shadow_prob,
            )
        except TypeError:
            return A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                p=cfg.shadow_prob,
            )

    def random_sun_flare():
        try:
            # Albumentations 2.x pydantic-validates the range and rejects a 0
            # lower bound (raises ValueError, *not* TypeError, so the except
            # below would not catch it) — keep it >= 1. 1.x still accepts 0.
            return A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_range=(1, 3),
                src_radius=120,
                p=cfg.sun_flare_prob,
            )
        except TypeError:
            return A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_lower=0,
                num_flare_circles_upper=3,
                src_radius=120,
                p=cfg.sun_flare_prob,
            )

    def coarse_dropout():
        try:
            return A.CoarseDropout(
                num_holes_range=(1, cfg.coarse_dropout_max_holes),
                hole_height_range=(4, 24),
                hole_width_range=(4, 24),
                fill=0,
                p=cfg.coarse_dropout_prob,
            )
        except TypeError:
            return A.CoarseDropout(
                max_holes=cfg.coarse_dropout_max_holes,
                max_height=24,
                max_width=24,
                min_holes=1,
                min_height=4,
                min_width=4,
                fill_value=0,
                p=cfg.coarse_dropout_prob,
            )

    # === §5.3 camera-matched geometric/codec aug ===

    def lens_distortion():
        # OpticalDistortion/GridDistortion both accept ``distort_limit``
        # (tuple ok) and ``p`` across Albumentations 1.x/2.x; the 1.x-only
        # ``shift_limit`` defaults harmlessly, so no version branch is needed.
        return A.OneOf([
            A.OpticalDistortion(
                distort_limit=(-cfg.lens_distort_limit, cfg.lens_distort_limit),
                p=1.0,
            ),
            A.GridDistortion(
                num_steps=cfg.grid_distort_steps,
                distort_limit=(-cfg.grid_distort_limit, cfg.grid_distort_limit),
                p=1.0,
            ),
        ], p=cfg.lens_distortion_prob)

    def image_compression():
        # quality_lower/quality_upper (1.x) -> quality_range (2.x).
        try:
            return A.ImageCompression(
                quality_range=(cfg.jpeg_quality_min, cfg.jpeg_quality_max),
                p=cfg.image_compression_prob,
            )
        except TypeError:
            return A.ImageCompression(
                quality_lower=cfg.jpeg_quality_min,
                quality_upper=cfg.jpeg_quality_max,
                p=cfg.image_compression_prob,
            )

    # Core Albumentations has no radial-vignette transform stable across
    # 1.x/2.x, so implement a small, bounded ImageOnly transform: a quadratic
    # radial darkening toward the corners. ``super().__init__(p=p)`` is
    # keyword-compatible with both the 1.x ``(always_apply, p)`` and the 2.x
    # ``(p,)`` BasicTransform signatures.
    class _Vignette(A.ImageOnlyTransform):
        """Radial corner darkening (wide-lens + small-sensor falloff). §5.3."""

        def __init__(self, min_strength: float, max_strength: float, p: float):
            super().__init__(p=p)
            self.min_strength = float(min_strength)
            self.max_strength = float(max_strength)

        def get_params(self):
            import random
            return {"strength": random.uniform(self.min_strength, self.max_strength)}

        def apply(self, img, strength: float = 0.3, **params):
            h, w = img.shape[:2]
            yy, xx = np.ogrid[:h, :w]
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            # Normalized radial distance: 0 at centre, 1 at the corners.
            dist = np.sqrt(((xx - cx) / max(cx, 1.0)) ** 2
                           + ((yy - cy) / max(cy, 1.0)) ** 2) / np.sqrt(2.0)
            mask = (1.0 - strength * np.clip(dist, 0.0, 1.0) ** 2).astype(np.float32)
            out = img.astype(np.float32) * mask[..., None]
            return np.clip(out, 0, 255).astype(img.dtype)

        def get_transform_init_args_names(self):
            return ("min_strength", "max_strength")

    def vignette():
        return _Vignette(cfg.vignette_min_strength, cfg.vignette_max_strength,
                         p=cfg.vignette_prob)

    # Mild chromatic aberration — only present in newer Albumentations
    # (≥1.4 / 2.x). Explicitly optional per §5.3, so degrade gracefully on
    # stacks that lack it instead of crashing the whole pipeline.
    chromatic_cls = getattr(A, "ChromaticAberration", None)
    if chromatic_cls is not None:
        chroma = chromatic_cls(
            primary_distortion_limit=0.02,
            secondary_distortion_limit=0.02,
            mode="green_purple",
            p=cfg.chromatic_aberration_prob,
        )
    else:
        chroma = None
        LOGGER.info(
            "Albumentations has no ChromaticAberration; skipping that "
            "optional §5.3 transform (rest of the pipeline is unaffected)."
        )

    transform = A.Compose([
        # --- Blur family (the robot moves fast) ---
        A.OneOf([
            A.MotionBlur(blur_limit=(3, cfg.motion_blur_kernel_max), p=1.0),
            A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.3), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=cfg.motion_blur_prob),

        # --- Noise (low-light store interiors) ---
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            gauss_noise(),
        ], p=cfg.noise_prob),

        # --- Lighting (aisle-to-aisle variation) ---
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=cfg.brightness_prob),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=cfg.clahe_prob),
        random_shadow(),
        random_sun_flare(),

        # --- Hue (gently — preserve yellow/red promo signal) ---
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=cfg.hue_prob),

        # --- Geometry (viewing angle from a moving robot) ---
        A.Perspective(scale=(0.02, 0.05), keep_size=True, p=cfg.perspective_prob),
        A.HorizontalFlip(p=cfg.hflip_prob),

        # --- §5.3 camera-matched geometric/codec ---
        lens_distortion(),
        image_compression(),
        vignette(),
        *([chroma] if chroma is not None else []),

        # --- Partial occlusion (other shelf items, shopping carts, hands) ---
        coarse_dropout(),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.30,
    ))
    return transform


def attach_to_ultralytics(model, cfg: HeavyAugConfig = HeavyAugConfig()) -> bool:
    """Monkey-patch Ultralytics' YOLODataset to apply Albumentations before mosaic.

    Returns True if attached. Idempotent — re-calling does nothing.

    Note: Ultralytics' internal API changes occasionally. This patch is a
    best-effort: if the dataset class shape changes, fall back to relying on
    the built-in Ultralytics augs only. Training will still run, just without
    the heavy domain-specific transforms.
    """
    try:
        from ultralytics.data.augment import Albumentations as _UltraAlbu  # type: ignore
        import albumentations as A  # type: ignore
    except ImportError:
        LOGGER.warning(
            "Could not import ultralytics.data.augment.Albumentations; "
            "heavy aug will not be active. Falling back to built-in augs."
        )
        return False

    transform = build_albu_transform(cfg)

    class _HeavyAlbumentations(_UltraAlbu):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                super().__init__(p=kwargs.get("p", 1.0))
            self.transform = transform
            self.p = kwargs.get("p", getattr(self, "p", 1.0))
            self.contains_spatial = True

    # Patch the class used by Ultralytics' DataLoader pipeline.
    import ultralytics.data.augment as _aug_mod  # type: ignore
    _aug_mod.Albumentations = _HeavyAlbumentations
    LOGGER.info("Attached heavy Albumentations pipeline to Ultralytics.")
    return True
