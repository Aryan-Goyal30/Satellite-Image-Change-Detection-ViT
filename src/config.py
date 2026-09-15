"""Central configuration: every project path in one place.

Before this module the same constants (repo root, tile root, outputs, checkpoint
name) were recomputed in train.py, evaluate.py, evaluate_baseline.py, figures.py,
compare_figures.py, engine.py, predict.py and app.py. Adding a second domain
would have multiplied them again.

Paths are plain strings, exactly as the previous module-level constants were, so
call sites keep using os.path.join / os.path.exists unchanged.
"""
import os

# repo root = parent of the src/ package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- data -------------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data")
LEVIR_RAW = os.path.join(DATA_DIR, "levir_cd")             # A/ B/ label/ + raw/
LEVIR_TILES = os.path.join(DATA_DIR, "levir_cd_tiles")     # {train,val,test}/{A,B,label}

# S2Looking (Stage 3B directionality study). Acquisition + label generation
# only: no production inference reads these paths. Imagery and label rasters
# live under data/ and are never committed.
S2LOOKING_RAW = os.path.join(DATA_DIR, "s2looking")        # layout verified after extraction
S2LOOKING_META = os.path.join(S2LOOKING_RAW, "metadata")   # region_labels.csv, dataset_stats.json
DEMO_IMAGES = os.path.join(ROOT, "images")                 # bundled sample pairs

# --- bundled example catalogue ----------------------------------------------
# The current input source. A future imagery provider (location + date) will
# occupy the same role; applications resolve inputs through this, never by
# knowing a dataset's directory layout.
EXAMPLES_DIR = os.path.join(ROOT, "examples")
EXAMPLES_CATALOGUE = os.path.join(EXAMPLES_DIR, "catalogue.json")

# --- analysis defaults ------------------------------------------------------
# Connected components smaller than this are discarded as noise. Single source
# of truth: the engine, the CLI and the UI all read it from here.
DEFAULT_MIN_AREA_PX = 32

# --- model ------------------------------------------------------------------
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINTS, "siamese_unet_r34_best.pt")
DEFAULT_CHECKPOINT_REL = os.path.relpath(DEFAULT_CHECKPOINT, ROOT)

# Direction classifier (Stage 3B). An OPTIONAL capability: the built-environment
# engine analyses without it and only loads it when direction is requested. The
# "both" (6-channel BEFORE+AFTER) checkpoint is the only one used in production;
# the before-only / after-only checkpoints are ablations.
DIRECTION_CHECKPOINT = os.path.join(CHECKPOINTS, "direction_resnet18_both_best.pt")
DIRECTION_CHECKPOINT_REL = os.path.relpath(DIRECTION_CHECKPOINT, ROOT)

# --- outputs ----------------------------------------------------------------
OUTPUTS = os.path.join(ROOT, "outputs")
RESULTS = os.path.join(OUTPUTS, "results")
FIGURES = os.path.join(OUTPUTS, "figures")
PREDICTIONS = os.path.join(OUTPUTS, "predictions")
BASELINE_OUTPUTS = os.path.join(OUTPUTS, "baseline")
LOGS = os.path.join(ROOT, "logs")

EVALUATION_JSON = os.path.join(RESULTS, "evaluation.json")
BASELINE_EVALUATION_JSON = os.path.join(RESULTS, "baseline_evaluation.json")

# --- environment domain (Phase 4B) ------------------------------------------
# The frozen E2 six-band baseline, selected on the v24 validation split. It
# lives under outputs/ rather than checkpoints/ because that is where the Phase
# 4B experiment wrote it, and the experiment manifest, training history and
# evaluation sitting beside it are what make the weights interpretable. Moving
# the file would break that link to its own provenance.
ENVIRONMENT_BASELINE_DIR = os.path.join(OUTPUTS, "environment_baseline")
ENVIRONMENT_CHECKPOINT = os.path.join(ENVIRONMENT_BASELINE_DIR,
                                      "environment_sixband_best.pt")
ENVIRONMENT_CHECKPOINT_REL = os.path.relpath(ENVIRONMENT_CHECKPOINT, ROOT)
ENVIRONMENT_EVALUATION_JSON = os.path.join(ENVIRONMENT_BASELINE_DIR, "evaluation.json")


def resolve(path, base=ROOT):
    """Absolute path for a value that may be given relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(base, path)
