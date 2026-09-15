"""Model card for the direction classifier (construction vs demolition).

Everything the system is allowed to claim about this model lives here rather
than as string literals inside inference code, mirroring how the detector's card
works (src/domains/built_environment/model_card.py).

This module holds constants and provenance only. It loads no weights, imports no
dataset code, and has no Streamlit dependency.

Full narrative card: docs/DIRECTION_CLASSIFIER_MODEL_CARD.md
"""
from src.domains.built_environment.direction.crops import (
    CONTEXT_FRACTION, CROP_SIZE, MIN_SIDE)

# ------------------------------------------------------------------ identity
MODEL_NAME = "direction-resnet18-6ch"
VERSION = "1.0.0"
ARCHITECTURE = "ResNet-18"
TASK = "Region-level change direction classification"
MODE = "both"                       # 6-channel BEFORE+AFTER; the production model
CHECKPOINT_NAME = "direction_resnet18_both_best.pt"

INPUT_CHANNELS = 6
INPUT_SIZE = (CROP_SIZE, CROP_SIZE)
CLASS_NAMES = ("construction", "demolition")
ABSTAIN_LABEL = "uncertain"
DIRECTION_VALUES = CLASS_NAMES + (ABSTAIN_LABEL,)

# ------------------------------------------------------------- crop protocol
# Re-exported from the canonical protocol module so there is exactly one source.
CROP_SIZE = CROP_SIZE
CONTEXT_FRACTION = CONTEXT_FRACTION
MIN_SIDE = MIN_SIDE

# --------------------------------------------------------------- abstention
#: Operating threshold, selected on the VALIDATION split only (largest tau still
#: reaching 90% validation coverage). The test split was never used to choose it.
DIRECTION_TAU = 0.960

ABSTENTION = (
    "UNCERTAIN is an abstention, not a trained class. The label space is exactly "
    "{construction, demolition}; no region is labelled 'uncertain' in any "
    "training or evaluation data. When the score for the predicted class falls "
    "below DIRECTION_TAU the direction is reported as 'uncertain', and the score "
    "is still reported so the reason is visible."
)

#: The single most important caveat about direction_score.
NOT_CALIBRATED = (
    "direction_score is the model's score for the predicted class. It is NOT a "
    "calibrated probability: on the S2Looking test split mean confidence was "
    "0.9770 against accuracy 0.9631 (ECE 0.0172), i.e. the model is measurably "
    "over-confident. Present it as a score, or calibrate it first."
)

# ------------------------------------------------------------------- dataset
DATASET = "S2Looking"
REGION_COUNTS = {
    "train": {"regions": 26743, "construction": 17016, "demolition": 9727},
    "val": {"regions": 4001, "construction": 2730, "demolition": 1271},
    "test": {"regions": 7617, "construction": 4726, "demolition": 2891},
    "mixed_excluded_total": 1860,
}

EVALUATION_PROTOCOL = (
    "Official S2Looking scene-level split (3500/500/1000 scenes). Regions come "
    "from the Stage 3B-1 targets, produced by the production region extractor. "
    "MIXED regions were excluded from training and evaluation. Checkpoint "
    "selected on validation macro-F1; the abstention threshold was selected on "
    "validation only; the test split was untouched until final evaluation."
)

# ------------------------------------------------------------------- metrics
#: S2Looking test split, 7,617 ground-truth regions.
TEST_METRICS = {
    "regions": 7617,
    "macro_f1": 0.9608,
    "balanced_accuracy": 0.9601,
    "accuracy": 0.9631,
    "ece": 0.0172,
}

#: Same split, restricted to regions retained at DIRECTION_TAU.
RETAINED_AT_TAU = {
    "threshold": DIRECTION_TAU,
    "coverage": 0.8948,
    "accuracy": 0.9868,
    "macro_f1": 0.9858,
}

# -------------------------------------------------------- temporal semantics
#: Ratified PROJECT decision, not an author-documented property of S2Looking.
TEMPORAL_ORDERING = {
    "before_image_dir": "Image2",
    "after_image_dir": "Image1",
    "construction_label_dir": "label1",
    "demolition_label_dir": "label2",
    "basis": "Ratified project decision. Official S2Looking documentation does "
             "NOT state which of Image1/Image2 is earlier; the paper fixes only "
             "that label1 = newly built and label2 = demolished. See "
             "docs/S2LOOKING_TEMPORAL_SEMANTICS.md.",
    "documented_by_authors": False,
}

# --------------------------------------------------------------- limitations
NOT_CAPABLE_OF = (
    "reporting a calibrated probability for a direction",
    "classifying MIXED regions, where construction and demolition co-occur in "
    "one component - these were excluded from training and evaluation",
    "identifying or counting individual buildings",
    "operating outside its validated envelope (see OPERATING_ENVELOPE)",
)

OPERATING_ENVELOPE = (
    "Trained and evaluated on S2Looking: rural, globally distributed, "
    "off-nadir (mean |angle| ~9.9 deg, max 35.4 deg), 0.5-0.8 m GSD. The "
    "built-environment detector is trained on LEVIR-CD: urban, near-nadir, "
    "0.5 m. Applying the classifier to LEVIR-like imagery is OUT OF ENVELOPE "
    "and the reported metrics do not transfer."
)

#: Evaluated on ground-truth regions, never end to end.
EVALUATION_SCOPE = (
    "All reported metrics are for classification of S2Looking GROUND-TRUTH "
    "regions. The classifier has NOT been validated end to end on regions "
    "produced by the frozen LEVIR detector; such a result would confound "
    "detector domain shift with classifier quality and has not been measured."
)


def provenance(weights_hash=None, checkpoint=None) -> dict:
    """Direction provenance, as carried on a ChangeResult when direction is active.

    Small and flat by design: the core `Provenance` dataclass describes exactly
    one model (the detector), so this travels alongside it rather than
    overloading it. See the Stage 3B-3 report for the reasoning.
    """
    return {
        "model": MODEL_NAME,
        "version": VERSION,
        "architecture": ARCHITECTURE,
        "task": TASK,
        "dataset": DATASET,
        "checkpoint": checkpoint,
        "weights_hash": weights_hash,
        "threshold": DIRECTION_TAU,
        "threshold_selected_on": "validation",
        "classes": list(CLASS_NAMES),
        "abstain_label": ABSTAIN_LABEL,
        "crop_protocol": {
            "crop_size": CROP_SIZE,
            "context_fraction": CONTEXT_FRACTION,
            "min_side_px": MIN_SIDE,
            "channels": "0:3 = BEFORE RGB, 3:6 = AFTER RGB",
        },
        "temporal_ordering": dict(TEMPORAL_ORDERING),
        "documented_by_authors": TEMPORAL_ORDERING["documented_by_authors"],
        "score_is_calibrated": False,
        "notes": NOT_CALIBRATED,
        "evaluation_scope": EVALUATION_SCOPE,
        "operating_envelope": OPERATING_ENVELOPE,
    }
