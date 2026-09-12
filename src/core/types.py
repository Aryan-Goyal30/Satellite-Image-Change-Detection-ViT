"""Earth Guardian core result contract.

Domain-independent. Nothing here imports torch, a dataset, or a model, so every
future domain engine (environmental, disaster) can produce the same result type
without depending on the Built Environment implementation.

    ChangeResult
        schema_version
        layers[]      one per detected change type; today exactly one
        regions[]     connected components of a layer
        quantities    pixels, percentage, and m2 ONLY when scale is known
        provenance    model, weights hash, dataset, threshold, protocol, envelope

Area rule (enforced, not merely documented): `area_m2` can only be produced by
Quantities.from_mask() when a GeoRef with a real ground sample distance is
supplied. There is no code path that fabricates it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# 1.1 adds optional fields only: GeoRef.units / GeoRef.pixel_size and
# Region.centroid_crs_xy. Every 1.0 field keeps its name and meaning, so a 1.0
# consumer reading a 1.1 result sees exactly what it saw before.
SCHEMA_VERSION = "1.1"


# --------------------------------------------------------------------- geo
@dataclass(frozen=True)
class GeoRef:
    """Optional geospatial reference for an acquisition.

    Everything is optional because the current data (LEVIR-CD PNGs) has none.
    `gsd_m` is the only field required to convert pixels into ground area, and
    it is set only when the source declares a projected CRS whose linear unit is
    metre - see src/common/georef.py. `units` records what the pixel size is
    measured in, so "degrees" can be reported without ever being treated as
    metres.
    """
    crs: Optional[str] = None
    transform: Optional[Sequence[float]] = None
    gsd_m: Optional[float] = None
    units: Optional[str] = None
    pixel_size: Optional[Sequence[float]] = None

    @property
    def has_scale(self) -> bool:
        return self.gsd_m is not None and self.gsd_m > 0

    def to_dict(self) -> dict:
        return {"crs": self.crs,
                "transform": list(self.transform) if self.transform else None,
                "gsd_m": self.gsd_m,
                "units": self.units,
                "pixel_size": list(self.pixel_size) if self.pixel_size else None}


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True)
class InputSpec:
    """What an engine requires of its input imagery.

    `in_channels` / `bands` exist so a future multispectral or SAR engine can
    declare its own requirements. The current Built Environment engine is and
    stays 3-channel RGB.
    """
    input_size: tuple = (256, 256)
    in_channels: int = 3
    channel_order: str = "RGB"
    bands: tuple = ("R", "G", "B")
    mean: Optional[tuple] = None
    std: Optional[tuple] = None

    def to_dict(self) -> dict:
        return {"input_size": list(self.input_size), "in_channels": self.in_channels,
                "channel_order": self.channel_order, "bands": list(self.bands),
                "mean": list(self.mean) if self.mean is not None else None,
                "std": list(self.std) if self.std is not None else None}


# ------------------------------------------------------------------ layers
@dataclass
class Layer:
    """One change type. Today there is exactly one: binary structural change.

    A semantic or multi-class engine would return several layers; that is the
    reason this is a list rather than a single mask.
    """
    name: str
    mask: np.ndarray                          # bool, H x W
    score_map: Optional[np.ndarray] = None    # float in [0, 1], H x W
    threshold: Optional[float] = None
    mean_confidence: Optional[float] = None   # mean score inside the mask
    description: str = ""

    @property
    def changed_pixels(self) -> int:
        return int(self.mask.sum())

    def to_dict(self) -> dict:
        """JSON-safe summary. Arrays stay out of the dict by design."""
        return {"name": self.name, "description": self.description,
                "threshold": self.threshold, "changed_pixels": self.changed_pixels,
                "mean_confidence": self.mean_confidence,
                "has_score_map": self.score_map is not None}


# ----------------------------------------------------------------- regions
@dataclass
class Region:
    """One connected component of a layer's change mask.

    An area of detected change - not an identified building, and with no
    direction (construction vs demolition) attached.

    `confidence` is the mean predicted change probability over the region's
    pixels; see src/common/regions.py for the exact definition. `area_m2` and
    `centroid_crs_xy` are populated only when a GeoRef with a metric scale /
    transform is available.
    """
    id: int
    area_px: int
    bbox_xywh: Sequence[int]
    centroid_xy: Sequence[float]
    confidence: Optional[float] = None
    layer: Optional[str] = None
    area_m2: Optional[float] = None
    centroid_crs_xy: Optional[Sequence[float]] = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "area_px": self.area_px,
             "bbox_xywh": list(self.bbox_xywh),
             "centroid_xy": list(self.centroid_xy)}
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.layer is not None:
            d["layer"] = self.layer
        if self.area_m2 is not None:
            d["area_m2"] = self.area_m2
        if self.centroid_crs_xy is not None:
            d["centroid_crs_xy"] = list(self.centroid_crs_xy)
        return d


# -------------------------------------------------------------- quantities
@dataclass
class Quantities:
    changed_pixels: int
    total_pixels: int
    changed_percentage: float
    area_m2: Optional[float] = None

    @classmethod
    def from_mask(cls, mask: np.ndarray, georef: Optional[GeoRef] = None) -> "Quantities":
        changed = int(mask.sum())
        total = int(mask.size)
        pct = round(100.0 * changed / total, 4) if total else 0.0
        # area_m2 is produced here and nowhere else, and only with a real GSD.
        area = round(changed * georef.gsd_m * georef.gsd_m, 1) \
            if (georef is not None and georef.has_scale) else None
        return cls(changed_pixels=changed, total_pixels=total,
                   changed_percentage=pct, area_m2=area)

    def to_dict(self) -> dict:
        return {"changed_pixels": self.changed_pixels,
                "total_pixels": self.total_pixels,
                "changed_percentage": self.changed_percentage,
                "area_m2": self.area_m2}


# -------------------------------------------------------------- provenance
@dataclass(frozen=True)
class OperatingEnvelope:
    """Where an engine has actually been validated. Claims stop here."""
    gsd_m_range: Optional[tuple] = None
    regions_validated: tuple = ()
    sensors_validated: tuple = ()
    notes: str = ""

    def to_dict(self) -> dict:
        return {"gsd_m_range": list(self.gsd_m_range) if self.gsd_m_range else None,
                "regions_validated": list(self.regions_validated),
                "sensors_validated": list(self.sensors_validated),
                "notes": self.notes}


@dataclass
class Provenance:
    model: str
    dataset: str
    threshold: float
    task: Optional[str] = None
    version: Optional[str] = None
    weights_hash: Optional[str] = None          # sha256 of the actual checkpoint
    evaluation_protocol: Optional[str] = None
    operating_envelope: Optional[OperatingEnvelope] = None

    def to_dict(self) -> dict:
        return {"model": self.model, "version": self.version, "task": self.task,
                "dataset": self.dataset, "threshold": self.threshold,
                "weights_hash": self.weights_hash,
                "evaluation_protocol": self.evaluation_protocol,
                "operating_envelope": (self.operating_envelope.to_dict()
                                       if self.operating_envelope else None)}


# ------------------------------------------------------------------ result
@dataclass
class ChangeResult:
    layers: list
    regions: list
    quantities: Quantities
    provenance: Provenance
    input_info: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    runtime_seconds: Optional[float] = None
    georef: Optional[GeoRef] = None
    warnings: list = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def primary_layer(self) -> Layer:
        return self.layers[0]

    def layer(self, name: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.name == name), None)

    def to_dict(self) -> dict:
        """JSON-safe v1 representation. Masks and score maps are not included."""
        return {"schema_version": self.schema_version,
                "layers": [l.to_dict() for l in self.layers],
                "regions": [r.to_dict() for r in self.regions],
                "quantities": self.quantities.to_dict(),
                "provenance": self.provenance.to_dict(),
                "input": dict(self.input_info),
                "georef": self.georef.to_dict() if self.georef else None,
                "params": dict(self.params),
                "runtime_seconds": self.runtime_seconds,
                "warnings": list(self.warnings)}
