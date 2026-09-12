"""Compatibility shim - the Built Environment engine moved into its domain package.

New location:
    src.domains.built_environment.engine

Preferred access is through the registry, which does not bind callers to a
particular model implementation:

    from src.core import registry
    engine = registry.get("built_environment")
"""
from src.domains.built_environment.engine import (  # noqa: F401
    DEFAULT_CKPT,
    ChangeEngine,
    overlay,
)

__all__ = ["ChangeEngine", "overlay", "DEFAULT_CKPT"]
