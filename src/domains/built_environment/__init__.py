"""Built Environment domain: binary structural (building) change detection.

Operational engine, trained on LEVIR-CD. Registered as "built_environment".
"""
DOMAIN = "built_environment"


def build_engine(**kwargs):
    """Factory used by src.core.registry.

    Imported lazily so that importing the registry does not pull in torch.
    """
    from src.domains.built_environment.engine import ChangeEngine
    return ChangeEngine(**kwargs)
