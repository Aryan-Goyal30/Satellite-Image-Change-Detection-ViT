"""Engine registry: name -> engine, resolved lazily.

    from src.core import registry
    engine = registry.get("built_environment")

Applications ask for a domain by name instead of importing a model module, so a
future environmental or disaster engine can be added without touching the UI.

Resolution is lazy (by "module:attribute" string) so importing the registry does
not import torch or load any weights.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

# Built-in domains. Only the Built Environment engine exists today; the others
# are intentionally absent rather than stubbed.
_LAZY: Dict[str, str] = {
    "built_environment": "src.domains.built_environment:build_engine",
}

_FACTORIES: Dict[str, Callable[..., Any]] = {}


def register(name: str, factory: Callable[..., Any], *, overwrite: bool = False) -> None:
    """Register an engine factory under `name`."""
    if not overwrite and (name in _FACTORIES or name in _LAZY):
        raise ValueError(f"engine '{name}' is already registered")
    _FACTORIES[name] = factory


def register_lazy(name: str, target: str, *, overwrite: bool = False) -> None:
    """Register by import path, e.g. 'my.module:build_engine'."""
    if not overwrite and (name in _FACTORIES or name in _LAZY):
        raise ValueError(f"engine '{name}' is already registered")
    _LAZY[name] = target


def available() -> list:
    """Names that can be requested from this registry."""
    return sorted(set(_FACTORIES) | set(_LAZY))


def get_factory(name: str) -> Callable[..., Any]:
    if name in _FACTORIES:
        return _FACTORIES[name]
    if name in _LAZY:
        module_path, _, attr = _LAZY[name].partition(":")
        factory = getattr(importlib.import_module(module_path), attr)
        _FACTORIES[name] = factory
        return factory
    raise KeyError(f"unknown engine '{name}'. Available: {available()}")


def get(name: str, **kwargs) -> Any:
    """Instantiate the engine registered under `name`."""
    return get_factory(name)(**kwargs)
