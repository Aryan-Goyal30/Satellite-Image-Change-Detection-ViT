"""Model card for the Environment engine (Phase 4B E2, frozen).

Everything the system is allowed to claim about this model is declared here
rather than as strings inside the inference engine.

The threshold is NOT declared statically: it is read from the checkpoint at load
time (`threshold_hint`, the argmax-F1 operating point of a 201-point sweep over
the VALIDATION probability histogram, frozen before the test split was read), so
the card cannot drift from what inference actually uses. VALIDATED_THRESHOLD
below records the current checkpoint's value for documentation only.

Measured performance, frozen v24 TEST split (44 samples)
--------------------------------------------------------
    pixel      P 0.5505   R 0.6472   F1 0.5950   IoU 0.4235   AP 0.5670
    region     P 0.2500   R 0.0735   F1 0.1136   (IoU >= 0.25 matching)
    Amazon     pixel F1 0.5989 (37 samples)
    SE Asia    pixel F1 0.3636 (7 samples, 3 positive)

These are stated as they are. Pixel agreement is moderate; REGION agreement is
poor - the model recovers roughly the right total extent (predicted area 1.18x
ground truth) while disagreeing about how that extent is divided into distinct
events. A user must not read a detected region as a reliably identified
individual clearing. The SE Asia figure rests on three positive samples and is
an indication, not a measurement.

Phase 4B-4 tested whether supervising at TMF's native ~30 m support explains the
topology disagreement. It did not: the hypothesis was REFUTED, and the label
granularity explanation is accordingly rated LOW.
"""
from src.core.engine import EngineMetadata
from src.core.types import InputSpec, OperatingEnvelope, Provenance
from src.domains.environment import normalization
from src.domains.environment.inputs import REQUIRED_BANDS

DOMAIN = "environment"
DISPLAY_NAME = "Environment"
TASK = "Binary forest-loss change detection"
CAPABILITIES = ("forest_loss",)
DATASET = "TMF + Sentinel-2"
DATASET_VERSION = "v24"

#: Frozen dataset identity, so a result can be traced to the exact archive.
#: The manifest digest is a plain sha256 of manifest.json and is reproducible.
DATASET_MANIFEST_SHA256 = "23721c09c864297a7857a3576e4398f700e642c3ab1e4531462eb603a2f827d8"
#: Carried forward from the Phase 4B experiment manifests as RECORDED
#: provenance. Unlike the manifest digest it is not independently reproducible:
#: the rule that combined the 1,180 .npy files into one hash was never committed
#: to code, so this value can be compared between experiments but cannot be
#: recomputed from the archive. Worth fixing when the dataset is next touched.
DATASET_ARRAYS_SHA256 = "b53d4b3d89d111de653313bcaad9984c6bc5e240905771a5d09c5fa6cc32d248"

#: Scores are an operating point, not calibrated likelihoods. Never claim they
#: are: no calibration was fitted or measured at any point in Phase 4B.
SCORE_CALIBRATED = False

# What this engine must NOT be described as doing. Kept next to the claims so
# the two cannot drift apart.
NOT_CAPABLE_OF = (
    "detecting forest loss outside tropical moist forest",
    "operating in Africa, which is entirely absent from the training data",
    "distinguishing deforestation from degradation, logging roads or windthrow",
    "attributing a cause to the loss",
    "identifying or counting individual trees",
    "reliably delineating a clearing as one distinct event (region F1 0.1136)",
    "reporting a calibrated probability that a pixel is really forest loss",
    "dating the loss more precisely than the two supplied acquisitions",
    "flood, fire, snow/ice or structural change detection",
    "operating on RGB, three-band or four-band imagery",
    "retrieving imagery from a satellite provider",
)

# Documentation only - the live value comes from the checkpoint.
VALIDATED_THRESHOLD = 0.91

EVALUATION_PROTOCOL = (
    "Operating threshold selected by argmax F1 over a 201-point sweep of the "
    "v24 VALIDATION probability histogram and applied to the held-out TEST "
    "split unchanged; TP/FP/FN/TN aggregated globally over all pixels. Splits "
    "are separated by MGRS tile with a measured minimum separation of 100.58 km.")

INPUT_SPEC = InputSpec(
    input_size=(256, 256),
    in_channels=len(REQUIRED_BANDS),
    channel_order="B02,B03,B04,B08,B11,B12",
    bands=REQUIRED_BANDS,
    mean=tuple(normalization.MEAN),
    std=tuple(normalization.STD),
)

OPERATING_ENVELOPE = OperatingEnvelope(
    # Sentinel-2 L2A only. The model has never seen another resolution, and the
    # 10 m figure describes the imagery it was trained on - not an assumption
    # this engine makes about an arbitrary input.
    gsd_m_range=(10.0, 10.0),
    regions_validated=("Amazon basin (v24, 84.4% of samples)",
                       "Southeast Asia (v24, 15.6% of samples)"),
    sensors_validated=("Sentinel-2 L2A surface reflectance, six bands",),
    notes=("Detects TMF-labelled forest loss between two Sentinel-2 acquisitions "
           "taken one year either side of the event year, both inside a 1 Aug - "
           "15 Sep window. Validated ONLY on Amazon and Southeast Asian tropical "
           "moist forest; Africa is absent from the training data entirely and "
           "the model is not validated there. Requires surface reflectance for "
           "all six bands - RGB imagery is refused, never substituted. Assumes "
           "the two dates are already co-registered on one grid; no reprojection "
           "or resampling is performed. Pixel-level agreement is moderate "
           "(test F1 0.5950) and region-level agreement is poor (region F1 "
           "0.1136), so detected regions indicate where loss is likely, not how "
           "many distinct clearings occurred. Scores are uncalibrated."),
)

LAYER_NAME = "forest_loss"
LAYER_DESCRIPTION = "TMF-labelled forest loss between the two acquisitions"

DESCRIPTION = ("Six-band spectral Siamese U-Net (ResNet-34) trained on JRC "
               "Tropical Moist Forest labels and Sentinel-2 L2A imagery for "
               "pixel-level binary forest-loss detection.")


def build_metadata(model_name, version, threshold, weights_hash=None):
    """Compose engine metadata from the static card plus checkpoint facts."""
    provenance = Provenance(
        model=model_name,
        dataset=DATASET,
        dataset_version=DATASET_VERSION,
        threshold=threshold,
        task=TASK,
        version=version,
        weights_hash=weights_hash,
        evaluation_protocol=EVALUATION_PROTOCOL,
        operating_envelope=OPERATING_ENVELOPE,
        score_calibrated=SCORE_CALIBRATED,
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
        description=DESCRIPTION,
    )
