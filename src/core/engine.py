"""Earth Guardian engine contract.

An engine turns a pair of co-located images into a ChangeResult. The contract
exists so applications (CLI, Streamlit, a future API) can hold an engine without
knowing which neural network is behind it.

Implemented today:
    built_environment   Siamese U-Net (ResNet-34), binary structural change

Planned, NOT implemented:
    environmental       vegetation / land cover / water change
    disaster            flood, burn scar, landslide, post-disaster damage
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from src.core.types import ChangeResult, InputSpec, Provenance


@dataclass
class EngineMetadata:
    """What an engine is, and what it is allowed to claim."""
    name: str
    domain: str
    task: str
    capabilities: tuple                 # change types this engine can output
    input_spec: InputSpec
    provenance: Provenance
    version: str = "1.0.0"
    description: str = ""
    display_name: str = ""              # human-readable domain name for a UI
    limitations: tuple = ()             # what this engine must NOT be said to do

    def to_dict(self) -> dict:
        return {"name": self.name, "domain": self.domain,
                "display_name": self.display_name, "task": self.task,
                "version": self.version, "description": self.description,
                "capabilities": list(self.capabilities),
                "limitations": list(self.limitations),
                "input_spec": self.input_spec.to_dict(),
                "provenance": self.provenance.to_dict()}


@runtime_checkable
class ChangeEngineProtocol(Protocol):
    """Structural interface every domain engine satisfies.

    Deliberately small: a registry entry, a metadata block, and one call.
    """

    domain: str

    @property
    def metadata(self) -> EngineMetadata:
        ...

    def analyze(self, before: Any, after: Any, **kwargs) -> ChangeResult:
        ...
