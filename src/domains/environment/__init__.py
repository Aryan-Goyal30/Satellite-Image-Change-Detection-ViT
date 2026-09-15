"""Environment domain: TMF-labelled forest-loss change detection.

Operational engine, trained on JRC Tropical Moist Forest labels and Sentinel-2
L2A imagery (dataset v24). Registered as "environment".

Two parts share this package and should not be confused:

    data/            Phase 4A acquisition and QC - builds the frozen dataset.
                     Never touched by inference.
    model, engine    Phase 4B. The frozen E2 six-band baseline and the engine
                     that serves it.

The engine requires all six bands (B02, B03, B04, B08, B11, B12) as surface
reflectance and refuses RGB input rather than substituting for the missing SWIR
bands - see inputs.py.
"""
DOMAIN = "environment"


def build_engine(**kwargs):
    """Factory used by src.core.registry.

    Imported lazily so that importing the registry does not pull in torch.
    """
    from src.domains.environment.engine import ChangeEngine
    return ChangeEngine(**kwargs)
