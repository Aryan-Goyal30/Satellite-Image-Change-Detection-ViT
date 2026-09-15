"""Production inference wrapper for the direction classifier.

Turns an image pair plus detected regions into construction / demolition /
uncertain predictions. It owns loading and caching of the trained "both"
checkpoint and nothing else: the architecture comes from `model.py`, the
preprocessing from `crops.py`, and the claims from `model_card.py`.

Importing this module loads no weights. The checkpoint is read on first use and
then retained, so one engine instance loads the model at most once.

Direction is an OPTIONAL capability. Every failure to make it available is
raised as `DirectionUnavailable`, carrying a short user-facing message, so the
caller can surface a warning instead of losing the detector's result.
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch

from src import config
from src.common.model_loader import checkpoint_sha256
from src.domains.built_environment.direction import crops
from src.domains.built_environment.direction import model_card as card
from src.domains.built_environment.direction.model import build_model

#: Regions per forward pass. Crops are small (6x128x128), so this is modest.
DEFAULT_BATCH_SIZE = 32


class DirectionUnavailable(RuntimeError):
    """Direction classification could not be performed.

    `user_message` is a short, non-technical explanation suitable for a warning
    shown to a user. Tracebacks are never propagated to callers.
    """

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


class DirectionClassifier:
    """Lazy-loading wrapper around the trained 6-channel direction model."""

    def __init__(self, checkpoint=None, device=None):
        self.checkpoint_path = config.resolve(
            checkpoint or config.DIRECTION_CHECKPOINT_REL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._weights_hash = None

    # ------------------------------------------------------------- loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self):
        """Load once, then reuse. Raises DirectionUnavailable on any problem."""
        if self._model is not None:
            return self._model

        path = self.checkpoint_path
        shown = os.path.relpath(path, config.ROOT) if path.startswith(config.ROOT) else path
        if not os.path.exists(path):
            raise DirectionUnavailable(
                f"the direction model checkpoint is not present ({shown}).")

        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
        except Exception:
            raise DirectionUnavailable(
                f"the direction model checkpoint could not be read ({shown}).")

        # The production model is the 6-channel BEFORE+AFTER one. The
        # before-only / after-only ablations must never be served.
        mode = ck.get("mode")
        if mode != card.MODE:
            raise DirectionUnavailable(
                f"the checkpoint is a '{mode}' model, but production requires "
                f"the '{card.MODE}' (6-channel BEFORE+AFTER) model.")

        names = tuple(ck.get("class_names") or ())
        if names != card.CLASS_NAMES:
            raise DirectionUnavailable(
                f"the checkpoint declares classes {names or '(none)'}, expected "
                f"{card.CLASS_NAMES}.")

        try:
            model = build_model(card.MODE, pretrained=False).to(self.device)
            model.load_state_dict(ck["model_state"])
        except Exception:
            raise DirectionUnavailable(
                "the direction model checkpoint is not compatible with the "
                "expected architecture.")

        if model.conv1.in_channels != card.INPUT_CHANNELS:
            raise DirectionUnavailable(
                f"the loaded model takes {model.conv1.in_channels} input "
                f"channels, expected {card.INPUT_CHANNELS}.")

        model.eval()
        self._model = model
        self._weights_hash = checkpoint_sha256(path)
        return model

    # ---------------------------------------------------------- inference
    @torch.inference_mode()
    def classify_regions(self, before, after, regions: Sequence,
                         batch_size: int = DEFAULT_BATCH_SIZE
                         ) -> List[Tuple[str, float]]:
        """Predict a direction for each region.

        `before` / `after` are the engine's two input images, in the engine's
        own argument order - they are never swapped here. BEFORE occupies
        channels 0:3 and AFTER channels 3:6, which is the ordering the model was
        trained with under the project's temporal convention.

        Returns one ``(direction, direction_score)`` per region, in the same
        order. `direction` is "construction", "demolition", or "uncertain" when
        the score falls below the operating threshold; the score is reported
        either way, and is NOT a calibrated probability.
        """
        if not regions:
            return []
        model = self.load()

        out: List[Tuple[str, float]] = []
        for start in range(0, len(regions), batch_size):
            chunk = regions[start:start + batch_size]
            batch = torch.cat([
                crops.region_model_input(before, after, r.bbox_xywh, batched=True)
                for r in chunk]).to(self.device)
            probs = torch.softmax(model(batch).float(), dim=1)
            scores, indices = probs.max(dim=1)
            for score, index in zip(scores.tolist(), indices.tolist()):
                out.append(label_for(index, score))
        return out

    # -------------------------------------------------------- provenance
    def provenance(self) -> dict:
        """Direction provenance for the result, including the checkpoint hash."""
        path = self.checkpoint_path
        shown = os.path.relpath(path, config.ROOT) if path.startswith(config.ROOT) else path
        return card.provenance(weights_hash=self._weights_hash, checkpoint=shown)


def label_for(index: int, score: float) -> Tuple[str, float]:
    """Apply the abstention rule to one prediction.

    Pure and model-free, so the threshold behaviour is testable on its own.
    """
    label = (card.CLASS_NAMES[index] if score >= card.DIRECTION_TAU
             else card.ABSTAIN_LABEL)
    return label, round(float(score), 4)


__all__ = ["DirectionClassifier", "DirectionUnavailable", "label_for",
           "DEFAULT_BATCH_SIZE"]
