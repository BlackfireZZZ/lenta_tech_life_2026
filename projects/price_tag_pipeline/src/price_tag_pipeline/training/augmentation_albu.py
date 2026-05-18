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
            return A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                num_flare_circles_range=(0, 3),
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
