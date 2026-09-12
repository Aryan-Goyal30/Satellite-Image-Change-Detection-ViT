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
preprocessing contract, checkpoint loading from the shared model loader, and
region extraction from the shared region module.

Handles images larger than the model's 256x256 training tile by sliding-window
inference with Hann-weighted blending, so a full 1024x1024 scene comes back
without tile seams.

Area units
----------
Ordinary PNG/JPEG imagery carries no georeferencing, so this engine reports
PIXEL counts and PERCENTAGES only. `area_m2` is produced solely by
Quantities.from_mask() / extract_regions() when a GeoRef with a real metric
scale is supplied by the caller - see src/common/georef.py. We do not fabricate
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
from src.common.regions import extract_regions
from src.common.visualization import overlay  # noqa: F401  (re-export)
from src.core.types import ChangeResult, GeoRef, Layer, Quantities
from src.domains.built_environment import model_card

DEFAULT_CKPT = config.DEFAULT_CHECKPOINT_REL

# The legacy result dict is a published output shape (predict.py writes it to
# disk behind --legacy-json). It is frozen deliberately.
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

    # -------------------------------------------------------------- public
    def analyze(self, before, after, threshold=None,
                min_area_px=config.DEFAULT_MIN_AREA_PX, gsd_m=None, georef=None):
        """Analyze one image pair.

        Args:
            threshold:    decision threshold; defaults to the validated value
            min_area_px:  connected components smaller than this are discarded
            gsd_m:        metres per pixel, when the caller knows it
            georef:       a GeoRef from src.common.georef, for GeoTIFF inputs.
                          Geographic output (m2, CRS centroids) appears only
                          when one of these carries a real metric scale.

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

        # A caller-supplied GeoRef wins; gsd_m remains supported for callers
        # that know only the scale. Neither invents anything on its own.
        if georef is None and gsd_m:
            georef = GeoRef(gsd_m=float(gsd_m), units="metre")
        effective_gsd = georef.gsd_m if (georef and georef.has_scale) else None

        tau = self.threshold if threshold is None else float(threshold)
        prob = self.probability_map(a_np, b_np)
        mask = (prob >= tau)

        H, W = mask.shape
        total_px = H * W

        warnings = []
        try:
            mask, regions = extract_regions(
                mask, score_map=prob, min_area_px=min_area_px,
                georef=georef, layer=model_card.LAYER_NAME)
        except ImportError:
            regions = None
            warnings = ["opencv not installed - region counting disabled"]

        changed_px = int(mask.sum())
        conf = float(prob[mask].mean()) if changed_px else 0.0

        # ---- legacy view: unchanged shape and keys ----
        legacy_regions = [
            {"id": r.id, "area_px": r.area_px,
             "bbox_xywh": list(r.bbox_xywh), "centroid_xy": list(r.centroid_xy)}
            for r in (regions or [])
        ]
        result = {
            "model": {
                "name": self.model_name,
                "version": self.version,
                "trained_on": self.trained_on,
                "checkpoint": self.checkpoint_path,
                "val_f1": round(self.val_f1, 4),
                "capability": LEGACY_CAPABILITY,
            },
            "input": {"height": H, "width": W, "gsd_m": effective_gsd},
            "params": {"threshold": round(tau, 4), "min_area_px": min_area_px,
                       "tile": self.tile, "overlap": self.overlap},
            "summary": {
                "changed_pixels": changed_px,
                "total_pixels": total_px,
                "changed_area_pct": round(100.0 * changed_px / total_px, 4),
                "n_regions": len(regions) if regions is not None else None,
                "mean_confidence": round(conf, 4),
                # Populated only when a real scale is available.
                "changed_area_m2": (round(changed_px * effective_gsd * effective_gsd, 1)
                                    if effective_gsd else None),
            },
            "regions": legacy_regions,
            "runtime_seconds": round(time.time() - t0, 3),
        }
        if regions is None:
            result["warnings"] = warnings

        # ---- v1 contract view ----
        layer = Layer(name=model_card.LAYER_NAME, mask=mask, score_map=prob,
                      threshold=round(tau, 4), mean_confidence=round(conf, 4),
                      description=model_card.LAYER_DESCRIPTION)
        change_result = ChangeResult(
            layers=[layer],
            regions=list(regions or []),
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


# `overlay` now lives in the shared presentation layer. Re-exported here so
# existing callers (predict.py, the src.inference.engine shim) keep working.
__all__ = ["ChangeEngine", "overlay", "DEFAULT_CKPT"]
