"""Model card for the Built Environment engine.

Everything the system is allowed to claim about this model is declared here
rather than hard-coded as strings inside the inference engine.

The threshold is NOT declared statically: it is read from the checkpoint at load
time (`val_best_threshold`, selected on the validation split), so the card can
never drift from what inference actually uses. VALIDATED_THRESHOLD below records
the value of the current checkpoint for documentation only.
"""
from src.core.engine import EngineMetadata
from src.core.types import InputSpec, OperatingEnvelope, Provenance
from src.common.preprocessing import IMAGENET_MEAN, IMAGENET_STD

DOMAIN = "built_environment"
DISPLAY_NAME = "Built Environment"
TASK = "Binary structural change detection"
CAPABILITIES = ("structural_change",)
DATASET = "LEVIR-CD"

# What this engine must NOT be described as doing. Kept next to the claims so
# the two cannot drift apart.
NOT_CAPABLE_OF = (
    "distinguishing construction from demolition",
    "identifying or counting individual buildings",
    "reporting real-world ground area",
    "flood, fire, deforestation or snow/ice detection",
    "retrieving imagery from a satellite provider",
)

# Documentation only - the live value comes from the checkpoint.
VALIDATED_THRESHOLD = 0.625

EVALUATION_PROTOCOL = (
    "Operating threshold selected on the LEVIR-CD validation split and applied "
    "to the test split unchanged; TP/FP/FN/TN aggregated globally over all pixels."
)

INPUT_SPEC = InputSpec(
    input_size=(256, 256),
    in_channels=3,
    channel_order="RGB",
    bands=("R", "G", "B"),
    mean=tuple(float(v) for v in IMAGENET_MEAN),
    std=tuple(float(v) for v in IMAGENET_STD),
)

OPERATING_ENVELOPE = OperatingEnvelope(
    gsd_m_range=(0.3, 1.0),
    regions_validated=("Texas, USA (LEVIR-CD, 2002-2018)",),
    sensors_validated=("Google Earth optical RGB",),
    notes=("Detects building construction and demolition in co-located optical "
           "imagery. Does NOT detect floods, fires, deforestation or snow/ice "
           "change. Not validated on other regions, sensors or coarser "
           "resolutions. Assumes the two dates are already co-registered. "
           "The training task is symmetric under swapping the two dates, so the "
           "model does not distinguish construction from demolition."),
)

LAYER_NAME = "structural_change"
LAYER_DESCRIPTION = "Building construction or demolition between the two dates"


def build_metadata(model_name, version, threshold, weights_hash=None):
    """Compose engine metadata from the static card plus checkpoint facts."""
    provenance = Provenance(
        model=model_name,
        dataset=DATASET,
        threshold=threshold,
        task=TASK,
        version=version,
        weights_hash=weights_hash,
        evaluation_protocol=EVALUATION_PROTOCOL,
        operating_envelope=OPERATING_ENVELOPE,
    )
    return EngineMetadata(
        name=model_name,
        domain=DOMAIN,
        display_name=DISPLAY_NAME,
        task=TASK,
        capabilities=CAPABILITIES,
        limitations=NOT_CAPABLE_OF,
        input_spec=INPUT_SPEC,
        provenance=provenance,
        version=version,
        description="Siamese U-Net (ResNet-34) trained on LEVIR-CD for "
                    "pixel-level binary structural change detection.",
    )
