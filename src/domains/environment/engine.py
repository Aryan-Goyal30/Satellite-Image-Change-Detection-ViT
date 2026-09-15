"""Environment change-analysis engine: TMF-labelled forest loss.

Implements the Earth Guardian engine contract (src/core/engine.py) for the
second operational domain, wrapping the frozen Phase 4B E2 six-band baseline.

    from src.core import registry
    engine = registry.get("environment")
    out = engine.analyze(before, after)      # H x W x 6 reflectance, both dates
    out["change_result"]                     # typed ChangeResult

No new model, no new training, no new data. This module contains inference
plumbing only; every number it reports about the model comes from the checkpoint
or from src/domains/environment/model_card.py.

Six bands, always
-----------------
The contract is enforced in src/domains/environment/inputs.py and refuses RGB
input rather than fabricating the missing bands. See that module for why.

Area units
----------
This engine reports PIXEL counts and PERCENTAGES unless the caller supplies a
GeoRef or a ground sample distance. Sentinel-2 L2A is distributed on a 10 m
grid, and inputs.SENTINEL2_GSD_M records that, but the engine does NOT assume it
of an arbitrary array handed to analyze(): the array may be a subset, a
reprojection or a crop of something else, and silently multiplying pixel counts
by 100 m2 would be inventing geographic metadata. A caller who knows the grid
passes gsd_m=inputs.SENTINEL2_GSD_M, or a GeoRef read from a GeoTIFF by
src/common/georef.py, and then area_m2 appears - produced by
Quantities.from_mask() / extract_regions(), the only code paths allowed to
produce it.

Precision
---------
Inference runs in float32 on every device. The frozen Phase 4B metrics were
measured on a bfloat16 autocast path on CUDA; Phase 4B-3 measured the difference
between the two at roughly 1e-3 in pixel precision and recall. float32 is chosen
here because a product's output should not depend on which GPU ran it.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from src import config
from src.common.model_loader import checkpoint_sha256
from src.common.regions import extract_regions
from src.common.tiling import sliding_window_probability
from src.core.types import ChangeResult, GeoRef, Layer, Quantities
from src.domains.environment import inputs, model_card, normalization
from src.domains.environment.loading import load_environment_model

DEFAULT_CKPT = config.ENVIRONMENT_CHECKPOINT_REL

#: Attached to every result that contains regions. The region-level metric is
#: the model's weakest (test region F1 0.1136), and a consumer that renders
#: regions without this caveat would overstate what they mean.
REGION_CAVEAT = (
    "Regions indicate where forest loss is likely, not how many distinct "
    "clearings occurred: region-level agreement on the frozen test split is "
    "poor (region F1 0.11).")


class ChangeEngine:
    """Environment engine. Satisfies src.core.engine.ChangeEngineProtocol."""

    domain = model_card.DOMAIN

    #: There is no direction classifier for forest loss, and the training task
    #: gives no basis for one. Declared False rather than left undefined so an
    #: application reads a fact instead of inferring from an AttributeError.
    supports_direction = False

    def __init__(self, checkpoint=None, device=None, tile=256, overlap=64):
        # The engine resolves its own configured default, so applications can
        # call registry.get("environment") without knowing a path. An explicit
        # checkpoint is still accepted, for CLI use and testing.
        path = config.resolve(checkpoint or DEFAULT_CKPT)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Environment checkpoint not found: {path}\n"
                "This is the frozen Phase 4B E2 baseline; it is produced by "
                "python scripts/train_environment_baseline.py and is not "
                "downloaded automatically.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, ck = load_environment_model(path, self.device)

        self.bands = tuple(self.model.bands)
        if self.bands != inputs.REQUIRED_BANDS:
            raise ValueError(
                f"Checkpoint {path} was trained on bands {list(self.bands)}, but "
                f"this engine implements the six-band contract "
                f"{list(inputs.REQUIRED_BANDS)}. Refusing to serve a model whose "
                f"input contract differs from the one it is published under.")

        self.model_name = self.model.NAME
        self.version = ck.get("version", "1.0.0")
        self.trained_on = model_card.DATASET
        self.dataset_version = model_card.DATASET_VERSION
        self.checkpoint_path = os.path.relpath(path, config.ROOT)
        self.epoch = ck.get("epoch")
        # Operating threshold chosen on the v24 validation split during training
        # and frozen before the test split was read. Read from the checkpoint so
        # it cannot drift from what the reported metrics used.
        self.threshold = float(ck.get("threshold_hint", model_card.VALIDATED_THRESHOLD))
        self.val_f1 = float(ck.get("val_f1", float("nan")))
        self.tile = tile
        self.overlap = overlap
        self.weights_hash = checkpoint_sha256(path)

        # Train-only normalisation, as code rather than as a git-ignored data
        # file. The digest travels in params so a result can be checked against
        # the checkpoint's own recorded normalisation.
        self.normalization = normalization.statistics(self.bands)
        self._mean = np.asarray(self.normalization["mean"], dtype=np.float32)
        self._std = np.asarray(self.normalization["std"], dtype=np.float32)

        self._metadata = model_card.build_metadata(
            model_name=self.model_name, version=self.version,
            threshold=self.threshold, weights_hash=self.weights_hash)

    @property
    def metadata(self):
        """Engine identity, capabilities, input spec and provenance."""
        return self._metadata

    # ---------------------------------------------------------------- utils
    def _norm(self, arr):
        """H x W x 6 reflectance -> 1 x 6 x H x W normalised tensor.

        Same arithmetic as the Phase 4B training data loader, so the model is
        fed exactly what it was trained on.
        """
        x = (np.asarray(arr, dtype=np.float32) - self._mean) / self._std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def _predict_tile(self, tiles):
        """One window position: normalise, run the model, return probabilities."""
        ta = self._norm(tiles[0]).to(self.device)
        tb = self._norm(tiles[1]).to(self.device)
        logits = self.model(ta, tb)
        return torch.sigmoid(logits.float())[0, 0].cpu().numpy()

    def probability_map(self, before, after):
        """Full-resolution forest-loss probability map, seamlessly tiled.

        Windowing and Hann blending are the shared implementation in
        src/common/tiling.py; only normalisation and the forward pass are this
        domain's.
        """
        return sliding_window_probability([before, after], self._predict_tile,
                                          tile=self.tile, overlap=self.overlap)

    # -------------------------------------------------------------- public
    def analyze(self, before, after, threshold=None,
                min_area_px=config.DEFAULT_MIN_AREA_PX, gsd_m=None, georef=None):
        """Analyze one co-located Sentinel-2 pair.

        Args:
            before, after: six-band surface reflectance for the two dates, in
                           any form src/domains/environment/inputs.py accepts.
                           RGB input raises BandContractError; it is never
                           padded or substituted.
            threshold:     decision threshold; defaults to the checkpoint's
                           validated operating point.
            min_area_px:   connected components smaller than this are discarded.
            gsd_m:         metres per pixel, when the caller knows it. Not
                           assumed - see the module docstring.
            georef:        a GeoRef from src/common/georef.py, for GeoTIFF
                           inputs. Geographic output (m2, CRS centroids) appears
                           only when one of these carries a real metric scale.

        Returns a dict with:
            change_result  typed ChangeResult, the v1 contract
            probability, mask, before, after   arrays
        """
        t0 = time.time()
        a_np, b_np = inputs.check_pair(before, after)

        # A caller-supplied GeoRef wins; gsd_m remains supported for callers
        # that know only the scale. Neither invents anything on its own.
        if georef is None and gsd_m:
            georef = GeoRef(gsd_m=float(gsd_m), units="metre")
        effective_gsd = georef.gsd_m if (georef and georef.has_scale) else None

        tau = self.threshold if threshold is None else float(threshold)
        prob = self.probability_map(a_np, b_np)
        mask = (prob >= tau)

        H, W = mask.shape
        warnings = []
        try:
            mask, regions = extract_regions(
                mask, score_map=prob, min_area_px=min_area_px,
                georef=georef, layer=model_card.LAYER_NAME)
        except ImportError:
            regions = []
            warnings.append("opencv not installed - region counting disabled")
        if regions:
            warnings.append(REGION_CAVEAT)

        changed_px = int(mask.sum())
        conf = float(prob[mask].mean()) if changed_px else 0.0

        layer = Layer(name=model_card.LAYER_NAME, mask=mask, score_map=prob,
                      threshold=round(tau, 4), mean_confidence=round(conf, 4),
                      description=model_card.LAYER_DESCRIPTION)
        change_result = ChangeResult(
            layers=[layer],
            regions=list(regions),
            quantities=Quantities.from_mask(mask, georef),
            provenance=self._metadata.provenance,
            input_info={"height": H, "width": W, "bands": list(self.bands),
                        "gsd_m": effective_gsd},
            params={"threshold": round(tau, 4), "min_area_px": min_area_px,
                    "tile": self.tile, "overlap": self.overlap,
                    "precision": "float32",
                    "checkpoint": self.checkpoint_path,
                    "normalisation_subset_sha256":
                        self.normalization["subset_sha256"],
                    "dataset_manifest_sha256": model_card.DATASET_MANIFEST_SHA256,
                    "dataset_arrays_sha256": model_card.DATASET_ARRAYS_SHA256},
            runtime_seconds=round(time.time() - t0, 3),
            georef=georef,
            warnings=warnings,
        )

        return {"change_result": change_result, "probability": prob,
                "mask": mask, "before": a_np, "after": b_np}


__all__ = ["ChangeEngine", "DEFAULT_CKPT", "REGION_CAVEAT"]
