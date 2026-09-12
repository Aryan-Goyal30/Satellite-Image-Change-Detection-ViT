"""Earth Guardian core: domain-independent contracts.

    types     ChangeResult, Layer, Region, Quantities, Provenance, GeoRef, InputSpec
    engine    ChangeEngineProtocol, EngineMetadata
    registry  engine name -> engine implementation

Nothing in this package imports torch, a dataset or a model.
"""
from src.core.types import (  # noqa: F401
    SCHEMA_VERSION, ChangeResult, GeoRef, InputSpec, Layer, OperatingEnvelope,
    Provenance, Quantities, Region,
)

__all__ = ["SCHEMA_VERSION", "ChangeResult", "GeoRef", "InputSpec", "Layer",
           "OperatingEnvelope", "Provenance", "Quantities", "Region"]
