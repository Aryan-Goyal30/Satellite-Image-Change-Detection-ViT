"""Built Environment change-analysis engine.

Implements the Earth Guardian engine contract (src/core/engine.py) for the
first operational domain: pixel-level BINARY structural (building) change
between two co-located optical images.

    from src.core import registry
    engine = registry.get("built_environment")
    out = engine.analyze(before, after)
    out["change_result"]   # typed ChangeResult (v1 contract)
    out["result"]          # legacy dict, frozen for the existing CLI/UI

No UI dependency, and no dataset dependency: normalisation comes from the shared
preprocessing contract, and checkpoint loading from the shared model loader.

Handles images larger than the model's 256x256 training tile by sliding-window
inference with Hann-weighted blending, so a full 1024x1024 scene comes back
without tile seams.

Area units
----------
LEVIR-CD ships plain PNGs with no georeferencing and no documented GSD, so this
engine reports PIXEL counts and PERCENTAGES only. `area_m2` is produced solely by
Quantities.from_mask() when a real GeoRef scale is supplied. We do not fabricate
ground areas.
"""
import os
import time

import numpy as np
import torch
from PIL import Image

from src import config
from src.common.model_loader import checkpoint_sha256, load_model
from src.common.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_batched_tensor
from src.core.types import ChangeResult, GeoRef, Layer, Quantities, Region
from src.domains.built_environment import model_card

DEFAULT_CKPT = config.DEFAULT_CHECKPOINT_REL

# The legacy result dict is a published output shape (predict.py writes it to
# disk, app.py renders it). It is frozen deliberately: migrating consumers to
# ChangeResult.to_dict() is a separate, explicit step.
LEGACY_CAPABILITY = "structural / building change detection"


def _hann2d(size):
    """2D Hann window for seamless blending of overlapping tiles."""
    w = np.hanning(size + 2)[1:-1]          # drop the zero endpoints
    win = np.outer(w, w).astype(np.float32)
    return np.maximum(win, 1e-3)            # never exactly zero


class ChangeEngine:
    """Built Environment engine. Satisfies src.core.engine.ChangeEngineProtocol."""

    domain = model_card.DOMAIN

    def __init__(self, checkpoint=None, device=None, tile=256, overlap=64):
        # The engine resolves its own configured default, so applications can
        # call registry.get("built_environment") without knowing a path.
        # An explicit checkpoint is still accepted, for CLI use and testing.
        path = config.resolve(checkpoint or DEFAULT_CKPT)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\nTrain one first: python -m src.train.train")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck = load_model(path, self.device)

        self.model_name = ck.get("model_name", "siamese-unet")
        self.version = ck.get("version", "1.0.0")
        # Provenance comes from the model card and the checkpoint, not from
        # literals inside the inference code.
        self.trained_on = model_card.DATASET
        self.checkpoint_path = os.path.relpath(path, config.ROOT)
        # operating threshold chosen on the validation split during training
        self.threshold = float(ck.get("val_best_threshold", {}).get("threshold", 0.5))
        self.val_f1 = float(ck.get("val_metrics", {}).get("f1", float("nan")))
        self.tile = tile
        self.overlap = overlap
        self.weights_hash = checkpoint_sha256(path)

        self._metadata = model_card.build_metadata(
            model_name=self.model_name, version=self.version,
            threshold=self.threshold, weights_hash=self.weights_hash)

    @property
    def metadata(self):
        """Engine identity, capabilities, input spec and provenance."""
        return self._metadata

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _load(img):
        if isinstance(img, (str, os.PathLike)):
            return np.array(Image.open(img).convert("RGB"))
        if isinstance(img, Image.Image):
            return np.array(img.convert("RGB"))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        return arr[..., :3]

    def _norm(self, arr):
        # Identical arithmetic to before; the constants now come from the shared
        # preprocessing contract rather than from the LEVIR dataset module.
        return to_batched_tensor(arr, IMAGENET_MEAN, IMAGENET_STD)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def probability_map(self, a_np, b_np):
        """Full-resolution change probability map, seamlessly tiled."""
        H, W = a_np.shape[:2]
        t, ov = self.tile, self.overlap
        stride = t - ov

        # pad so every position is covered by a full tile
        ph = max(t, int(np.ceil(max(H - t, 0) / stride)) * stride + t)
        pw = max(t, int(np.ceil(max(W - t, 0) / stride)) * stride + t)
        # 'symmetric' rather than 'reflect': when the image is smaller than one
        # tile the pad width exceeds the dimension, which older numpy rejects in
        # reflect mode. The padded margin is cropped off again below, so the
        # choice only has to be safe, not principled.
        pad = ((0, ph - H), (0, pw - W), (0, 0))
        a_p = np.pad(a_np, pad, mode="symmetric")
        b_p = np.pad(b_np, pad, mode="symmetric")

        acc = np.zeros((ph, pw), dtype=np.float32)
        wsum = np.zeros((ph, pw), dtype=np.float32)
        win = _hann2d(t)

        rows = list(range(0, ph - t + 1, stride))
        cols = list(range(0, pw - t + 1, stride))
        for r in rows:
            for c in cols:
                ta = self._norm(a_p[r:r + t, c:c + t]).to(self.device)
                tb = self._norm(b_p[r:r + t, c:c + t]).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=(self.device == "cuda")):
                    logits = self.model(ta, tb)
                prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
                acc[r:r + t, c:c + t] += prob * win
                wsum[r:r + t, c:c + t] += win

        return (acc / np.maximum(wsum, 1e-6))[:H, :W]

    # ------------------------------------------------------- postprocessing
    @staticmethod
    def _regions(mask, min_area_px):
        """Connected components, filtered by area. Returns (labels, list-of-dict)."""
        try:
            import cv2
        except ImportError:
            return None, None
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, n):                       # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area_px:
                labels[labels == i] = 0
                continue
            x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                          int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            out.append({"id": len(out) + 1, "area_px": area,
                        "bbox_xywh": [x, y, w, h],
                        "centroid_xy": [round(float(cents[i][0]), 1),
                                        round(float(cents[i][1]), 1)]})
        out.sort(key=lambda d: -d["area_px"])
        for j, d in enumerate(out, 1):
            d["id"] = j
        return labels, out

    # -------------------------------------------------------------- public
    def analyze(self, before, after, threshold=None, min_area_px=32, gsd_m=None):
        """Analyze one image pair.

        Returns a dict with:
            result         legacy JSON-serialisable dict (frozen output shape)
            change_result  typed ChangeResult, the v1 contract
            probability, mask, before, after   arrays
        """
        t0 = time.time()
        a_np, b_np = self._load(before), self._load(after)
        if a_np.shape[:2] != b_np.shape[:2]:
            raise ValueError(
                f"Image sizes differ: {a_np.shape[:2]} vs {b_np.shape[:2]}. "
                "The two dates must cover the same extent at the same size.")

        tau = self.threshold if threshold is None else float(threshold)
        prob = self.probability_map(a_np, b_np)
        mask = (prob >= tau)

        H, W = mask.shape
        total_px = H * W
        changed_px = int(mask.sum())

        labels, regions = self._regions(mask, min_area_px)
        if regions is not None:
            mask = labels > 0                     # apply the min-area filter
            changed_px = int(mask.sum())

        conf = float(prob[mask].mean()) if changed_px else 0.0
        warnings = [] if regions is not None else \
            ["opencv not installed - region counting disabled"]

        # ---- legacy view: byte-for-byte the same shape as before ----
        result = {
            "model": {
                "name": self.model_name,
                "version": self.version,
                "trained_on": self.trained_on,
                "checkpoint": self.checkpoint_path,
                "val_f1": round(self.val_f1, 4),
                "capability": LEGACY_CAPABILITY,
            },
            "input": {"height": H, "width": W, "gsd_m": gsd_m},
            "params": {"threshold": round(tau, 4), "min_area_px": min_area_px,
                       "tile": self.tile, "overlap": self.overlap},
            "summary": {
                "changed_pixels": changed_px,
                "total_pixels": total_px,
                "changed_area_pct": round(100.0 * changed_px / total_px, 4),
                "n_regions": len(regions) if regions is not None else None,
                "mean_confidence": round(conf, 4),
                # Populated only when a real GSD is supplied. LEVIR-CD has none.
                "changed_area_m2": round(changed_px * gsd_m * gsd_m, 1) if gsd_m else None,
            },
            "regions": regions if regions is not None else [],
            "runtime_seconds": round(time.time() - t0, 3),
        }
        if regions is None:
            result["warnings"] = warnings

        # ---- v1 contract view ----
        georef = GeoRef(gsd_m=gsd_m) if gsd_m else None
        layer = Layer(name=model_card.LAYER_NAME, mask=mask, score_map=prob,
                      threshold=round(tau, 4), mean_confidence=round(conf, 4),
                      description=model_card.LAYER_DESCRIPTION)
        typed_regions = [
            Region(id=r["id"], area_px=r["area_px"], bbox_xywh=r["bbox_xywh"],
                   centroid_xy=r["centroid_xy"], layer=model_card.LAYER_NAME)
            for r in (regions or [])
        ]
        change_result = ChangeResult(
            layers=[layer],
            regions=typed_regions,
            quantities=Quantities.from_mask(mask, georef),
            provenance=self._metadata.provenance,
            input_info={"height": H, "width": W},
            params=dict(result["params"]),
            runtime_seconds=result["runtime_seconds"],
            georef=georef,
            warnings=warnings,
        )

        return {"result": result, "change_result": change_result,
                "probability": prob, "mask": mask, "before": a_np, "after": b_np}


def overlay(base_rgb, mask, color=(255, 40, 40), alpha=0.45):
    """Blend a binary mask over an image as a translucent colour layer."""
    out = base_rgb.astype(np.float32).copy()
    col = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * col
    return out.clip(0, 255).astype(np.uint8)
