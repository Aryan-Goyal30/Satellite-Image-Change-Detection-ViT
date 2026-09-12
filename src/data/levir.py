"""Compatibility shim - the LEVIR-CD dataset moved into its domain package.

New location:
    src.domains.built_environment.data.levir

This module re-exports the same names so existing imports keep working. It will
be removed once all call sites use the domain path.
"""
from src.domains.built_environment.data.levir import (  # noqa: F401
    IMAGENET_MEAN,
    IMAGENET_STD,
    LevirCDTiles,
    denormalize,
)

__all__ = ["LevirCDTiles", "denormalize", "IMAGENET_MEAN", "IMAGENET_STD"]
