"""Input validation for image pairs.

Pure checks against an engine's declared InputSpec. No UI framework is imported
here, so the Streamlit app, the CLI and any future frontend apply exactly the
same rules and produce the same messages.

The checks are deliberately conservative: they reject what the current engine
genuinely cannot accept (mismatched sizes, non-RGB-convertible data) and warn
about things that merely degrade quality.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError

# Modes PIL can convert to RGB without inventing data.
CONVERTIBLE_MODES = {"RGB", "RGBA", "L", "LA", "P", "I;16", "I", "F", "CMYK"}

# Below the model tile the engine pads; above this the run just gets slow.
TILE_HINT = 256
LARGE_IMAGE_HINT = 4096


@dataclass
class Issue:
    level: str      # "error" blocks analysis, "warning" does not
    message: str


@dataclass
class PairValidation:
    issues: list = field(default_factory=list)
    before_size: Optional[tuple] = None
    after_size: Optional[tuple] = None

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


def open_image(source):
    """Open an uploaded file or path as a PIL image.

    Raises ValueError with a user-facing message rather than a PIL traceback.
    """
    try:
        img = Image.open(source)
        img.load()
        return img
    except UnidentifiedImageError:
        raise ValueError("This file is not a readable image. Supported formats: "
                         "PNG, JPEG, TIFF.")
    except OSError as e:
        raise ValueError(f"The image could not be read: it may be truncated or "
                         f"corrupt ({e}).")


def _is_blank(img) -> bool:
    """True when an image carries no variation at all (uniform fill)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return arr.size == 0 or float(arr.std()) == 0.0


def validate_pair(before, after, input_spec=None) -> PairValidation:
    """Check a before/after pair against what the engine can accept."""
    v = PairValidation(before_size=before.size, after_size=after.size)
    channels = getattr(input_spec, "in_channels", 3) if input_spec else 3
    order = getattr(input_spec, "channel_order", "RGB") if input_spec else "RGB"

    # --- blocking: the two dates must describe the same extent -------------
    if before.size != after.size:
        v.issues.append(Issue("error",
            f"The two images must have the same dimensions. "
            f"BEFORE is {before.size[0]}x{before.size[1]}, "
            f"AFTER is {after.size[0]}x{after.size[1]}. "
            f"Crop or resample them to the same grid before analysing - "
            f"resizing one on its own would misalign the comparison."))

    # --- blocking: the engine is a 3-channel RGB model ----------------------
    for label, img in (("BEFORE", before), ("AFTER", after)):
        n_bands = len(img.getbands())
        if img.mode not in CONVERTIBLE_MODES:
            v.issues.append(Issue("error",
                f"{label}: image mode '{img.mode}' cannot be interpreted as "
                f"{channels}-channel {order}. This engine expects optical "
                f"{order} imagery."))
        elif n_bands > 4:
            v.issues.append(Issue("error",
                f"{label}: this image has {n_bands} bands. This engine is a "
                f"{channels}-channel {order} model and does not support "
                f"multispectral or SAR products."))
        elif img.mode in {"I;16", "I", "F"}:
            v.issues.append(Issue("warning",
                f"{label}: single-band {img.mode} data will be replicated across "
                f"{order} channels. Results may be unreliable."))

    # --- non-blocking quality hints ----------------------------------------
    for label, img in (("BEFORE", before), ("AFTER", after)):
        if min(img.size) < TILE_HINT:
            v.issues.append(Issue("warning",
                f"{label} is smaller than {TILE_HINT}x{TILE_HINT}; it will be "
                f"padded to the model tile and detail may be limited."))
        if max(img.size) > LARGE_IMAGE_HINT:
            v.issues.append(Issue("warning",
                f"{label} is {img.size[0]}x{img.size[1]}; large scenes take "
                f"noticeably longer to analyse."))
        try:
            if _is_blank(img):
                v.issues.append(Issue("error",
                    f"{label} appears to be blank (a single uniform colour)."))
        except Exception:
            pass

    return v


def to_rgb(img):
    """Convert to RGB the way the engine will, without inventing data."""
    return img if img.mode == "RGB" else img.convert("RGB")
