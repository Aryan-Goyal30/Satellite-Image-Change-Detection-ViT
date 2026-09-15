"""Domain-aware input preparation: sources in, engine arguments out.

Engines do not share an input contract. The built-environment detector reads
three-channel RGB images; the environment detector reads six Sentinel-2 bands as
surface reflectance and refuses anything else. Before this module, applications
resolved that difference by simply assuming RGB - `open_image` then `to_rgb` -
which is correct for exactly one domain.

    pair = pair_input.prepare(engine.domain, before, after, engine.metadata.input_spec)
    if not pair.ok:
        ...report pair.errors...
    out = engine.analyze(pair.before, pair.after, georef=pair.georef)

What this module is and is not
------------------------------
It DISPATCHES. It contains no normalisation, no band arithmetic, no model and no
knowledge of what any domain's imagery means. Each adapter delegates to code
that already owns the rules:

    built_environment   src/common/image_input.py   (unchanged, RGB + GeoTIFF)
    environment         src/domains/environment/inputs.py

Adapters are resolved lazily by import path, for the same reason the engine
registry is: naming a domain must not import that domain's dependencies.

Failure is a value, not an exception
------------------------------------
A pair that cannot be analysed comes back with `ok = False` and user-facing
messages in `errors`, because both callers - a CLI that must print and exit, and
a UI that must render a message - need the same text. Adapters that raise a
domain error (the environment band contract) have it translated here.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# domain -> "module:function". A future domain registers its own; nothing here
# is imported until that domain is actually asked for.
_ADAPTERS: Dict[str, str] = {
    "built_environment": "src.common.image_input:prepare_rgb_pair",
    "environment": "src.domains.environment.inputs:prepare_application_pair",
}


@dataclass
class PreparedPair:
    """Everything an application needs to call analyze() and show the result.

    `before` / `after` are whatever THIS domain's engine accepts, and are passed
    through untouched. `preview_before` / `preview_after` are uint8 RGB arrays
    for display and export, and are never what the model reads.
    """
    domain: str
    before: Any = None
    after: Any = None
    preview_before: Optional[Any] = None
    preview_after: Optional[Any] = None
    georef: Optional[Any] = None
    errors: list = field(default_factory=list)      # user-facing strings
    warnings: list = field(default_factory=list)    # user-facing strings
    notes: str = ""                                 # one-line situation summary
    validation: Optional[Any] = None                # domain-specific detail

    @property
    def ok(self) -> bool:
        return not self.errors and self.before is not None


def domains() -> list:
    """Domains this module can prepare input for."""
    return sorted(_ADAPTERS)


def register_adapter(domain: str, target: str, *, overwrite: bool = False) -> None:
    """Register an adapter by import path, e.g. 'my.module:prepare'."""
    if domain in _ADAPTERS and not overwrite:
        raise ValueError(f"input adapter for '{domain}' is already registered")
    _ADAPTERS[domain] = target


def get_adapter(domain: str):
    if domain not in _ADAPTERS:
        raise KeyError(
            f"no input adapter for domain '{domain}'. Known: {domains()}. "
            "Each domain declares how its imagery is read; there is no default, "
            "because assuming RGB is what this module exists to stop.")
    module_path, _, attr = _ADAPTERS[domain].partition(":")
    return getattr(importlib.import_module(module_path), attr)


def prepare(domain: str, before, after, input_spec=None) -> PreparedPair:
    """Read a before/after pair the way `domain` requires.

    Never raises for bad input: a rejected pair comes back with ok == False and
    the reason in `errors`.
    """
    adapter = get_adapter(domain)
    try:
        return adapter(before, after, input_spec)
    except Exception as exc:                                     # noqa: BLE001
        # A domain may raise its own contract error (e.g. the environment
        # six-band refusal). Its user-facing sentence is preserved exactly;
        # a bare exception degrades to its message rather than a traceback.
        message = getattr(exc, "user_message", None) or str(exc)
        return PreparedPair(domain=domain, errors=[message])
