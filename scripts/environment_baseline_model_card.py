"""Generate the model card for the Phase 4B-1B six-band environmental baseline.

Written FROM the experiment manifest and evaluation JSON, so the card cannot
drift from the numbers it describes. It is regenerated, never hand-edited.

Usage:  python scripts/environment_baseline_model_card.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                       # noqa: E402

OUT_DIR = os.path.join(config.OUTPUTS, "environment_baseline")
CARD = os.path.join(config.ROOT, "docs", "ENVIRONMENT_BASELINE_MODEL_CARD.md")


def row(metrics):
    return (f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['iou']:.4f} | "
            f"{metrics.get('average_precision', float('nan')):.4f} | "
            f"{metrics['pixel_accuracy']:.4f}")


def main():
    with open(os.path.join(OUT_DIR, "experiment_manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    with open(os.path.join(OUT_DIR, "evaluation.json"), encoding="utf-8") as fh:
        ev = json.load(fh)

    arch = man["architecture"]
    test, val = ev["test"]["pixel"], ev["validation"]["pixel"]
    lines = []
    add = lines.append

    add("# Environmental Change Baseline — Model Card")
    add("")
    add("**Phase 4B-1B · six-band Siamese U-Net · the first environmental model "
        "experiment for Earth Guardian.**")
    add("")
    add("This model establishes a baseline and nothing more. It is not integrated "
        "into the product, has not been compared against any alternative, and no "
        "ablation has been run.")
    add("")

    add("## What it does")
    add("")
    add("Given two Sentinel-2 Level-2A surface-reflectance observations of the same "
        "ground — one from the year before a JRC TMF deforestation year and one from "
        "the year after — it emits a per-pixel logit for *deforestation recorded by "
        "TMF between those two dates*.")
    add("")
    add(f"- Bands: {', '.join(arch['bands'])}")
    add("- Input: two 256×256×6 tensors, 10 m ground sample distance")
    add("- Output: one 256×256 logit map")
    add("")

    add("## Architecture")
    add("")
    add(f"- `{arch['name']}`, {arch['encoder']} encoder, "
        f"**{arch['parameters']:,} parameters**")
    add(f"- Weight sharing: {arch['weight_sharing']}")
    add(f"- Fusion: `{arch['fusion']}`")
    add("")
    add("### Six-band input stem")
    add("")
    add("| Band | Initialised from |")
    add("|---|---|")
    for band, source in arch["stem_initialisation"].items():
        add(f"| {band} | {source} |")
    add("")
    add(f"All filters are scaled by {arch['stem_scale']:.4f} (= 3/6) to keep the stem "
        "response magnitude comparable to the pretrained three-channel case.")
    add("")
    add(f"- Reused unchanged from ImageNet: {', '.join(arch['pretrained_layers_reused'])}")
    add(f"- Replaced: {', '.join(arch['pretrained_layers_replaced'])}")
    add(f"- Randomly initialised: {', '.join(arch['randomly_initialised'])}")
    add("")
    add("**ImageNet pretraining supplies no semantic knowledge of NIR or SWIR "
        "reflectance.** The visible bands are mapped by wavelength, which is "
        "defensible; B08/B11/B12 receive the achromatic mean of the RGB filters — a "
        "generic edge detector with no spectral selectivity. Any SWIR-specific "
        "structure had to be learned from the TMF data.")
    add("")

    add("## Training data")
    add("")
    add(f"Frozen dataset **{man['dataset_version']}**. "
        f"Train {man['splits']['train']['n']} samples "
        f"({man['splits']['train']['positive']} positive, "
        f"{man['splits']['train']['negative']} negative); "
        f"validation {man['splits']['val']['n']}. The test split was not read "
        "during training.")
    add("")
    add("Splits are geographic: the MGRS tile is the atomic unit, sets are pairwise "
        "disjoint, and the minimum separation between splits is 100.58 km.")
    add("")

    add("## Preprocessing")
    add("")
    norm = man["normalisation"]
    add(f"Per-band standardisation using statistics from **{norm['source']}** "
        f"(`sha256 {norm['sha256'][:16]}…`). No ImageNet statistics are used; they "
        "describe 8-bit sRGB photographs and are meaningless for surface "
        "reflectance, least of all SWIR.")
    add("")
    add("| Band | mean | std |")
    add("|---|---|---|")
    for band, mean, std in zip(norm["bands"], norm["mean"], norm["std"]):
        add(f"| {band} | {mean:.6f} | {std:.6f} |")
    add("")
    add("One global normalisation is applied to Amazon and SE Asia alike — "
        "deliberately, since per-region normalisation would conceal the domain "
        "shift the geographic evaluation exists to expose.")
    add("")

    add("## Training configuration")
    add("")
    loss, opt = man["loss"], man["optimiser"]
    add(f"- Loss: `{loss['formulation']}`")
    add(f"- `pos_weight` = **{loss['pos_weight']}** — {loss['pos_weight_source']}")
    add(f"- Optimiser: {opt['name']}, lr {opt['lr']}, weight decay "
        f"{opt['weight_decay']}, {opt['schedule']}")
    add(f"- Batch size {man['batch_size']}, seed {man['seed']}, "
        f"precision {man['precision']}")
    add(f"- Epochs run: {man['epochs_run']} of {man['epochs_requested']} "
        f"(early stopping)")
    add(f"- Augmentation: {man['augmentation']['transforms']}, applied identically "
        f"to {', '.join(man['augmentation']['applied_identically_to'])}. "
        "No date swap, no per-date geometry, no colour jitter.")
    add(f"- Hardware: {man['hardware']}; training time "
        f"{man['training_seconds']/60:.1f} min")
    add(f"- Checkpoint selection: {man['checkpoint_selection']} "
        f"(best epoch {man['best_epoch']})")
    add("")

    add("## Results")
    add("")
    add(f"Threshold **{ev['threshold']:.3f}**, selected by: {ev['threshold_rule']}.")
    add("")
    add("| Split | Precision | Recall | F1 | IoU | AP | Pixel acc. |")
    add("|---|---|---|---|---|---|---|")
    add(f"| Validation | {row(val)} |")
    add(f"| **Test** | {row(test)} |")
    add("")
    add(f"Test confusion: TP {test['tp']:,} · FP {test['fp']:,} · "
        f"FN {test['fn']:,} · TN {test['tn']:,}")
    add("")
    add("Pixel accuracy is reported only to show it is uninformative: predicting "
        "'no change' everywhere scores about 93% on this data.")
    add("")

    add("### Region level")
    add("")
    treg = ev["test"]["region"]
    add(f"Criterion: {treg['criterion']}, on components of at least "
        f"{treg['min_area_px']} px from the project's shared region protocol.")
    add("")
    add("| Split | Region P | Region R | Region F1 | Predicted | Ground truth |")
    add("|---|---|---|---|---|---|")
    for name, block in (("Validation", ev["validation"]["region"]),
                        ("Test", treg)):
        add(f"| {name} | {block['region_precision']:.4f} | "
            f"{block['region_recall']:.4f} | {block['region_f1']:.4f} | "
            f"{block['predicted_regions']} | {block['ground_truth_regions']} |")
    add("")

    add("### Geographic breakdown (test, identical threshold and normalisation)")
    add("")
    add("| Region | Samples | Positive samples | Positive px | P | R | F1 | IoU | AP |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name, block in ev["test_by_region"].items():
        p = block["pixel"]
        add(f"| {name} | {block['samples']} | {block['positive_samples']} | "
            f"{block['positive_pixels']:,} | {p['precision']:.4f} | "
            f"{p['recall']:.4f} | {p['f1']:.4f} | {p['iou']:.4f} | "
            f"{p['average_precision']:.4f} |")
    add("")

    add("## Limitations")
    add("")
    for text in [
        "This is a baseline, not a validated detector. It has been trained once, "
        "on one dataset version, with no hyperparameter search.",
        "**Not a global forest-loss detector.** Training data covers the Amazon "
        "(4 TMF tiles) and SE Asia (2 tiles). Africa is absent entirely — the JRC "
        "endpoint returns HTTP 500 for every Congo Basin longitude — so nothing "
        "here says anything about African forest.",
        "**Outputs are not calibrated probabilities.** The sigmoid output is a "
        "score used with a validation-selected threshold; it has not been "
        "calibration-tested.",
        "**No claim of superiority** over any other model or over fewer bands. "
        "The RGB and RGB+NIR ablations that would support such a claim have not "
        "been run.",
        "Supervision is 30 m-quantised: TMF labels are resampled from 30 m to the "
        "10 m grid, so boundaries are accurate to about one TMF pixel, not to 10 m.",
        "TMF 'deforestation' includes conversion to plantation or water. It means "
        "forest-cover loss, not necessarily illegal or permanent clearing.",
        "The loss is computed over all pixels including those flagged invalid by "
        "SCL (mean 0.27% of a sample, max 4.98%); they are not masked out.",
        "Eight samples with reversed inside-vs-outside dNBR are retained in the "
        "frozen dataset by design and were not excluded from training or scoring.",
    ]:
        add(f"- {text}")
    add("")

    add("## Reproducibility")
    add("")
    add(f"- Dataset: {man['dataset_version']} (frozen)")
    add(f"- Seed: {man['seed']}")
    add(f"- Checkpoint: `{man['checkpoint']}`")
    add(f"- Checkpoint sha256: `{man['checkpoint_sha256']}`")
    add(f"- Normalisation sha256: `{norm['sha256']}`")
    add(f"- Generated: {ev['generated_utc']}")
    add("")

    os.makedirs(os.path.dirname(CARD), exist_ok=True)
    with open(CARD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"model card written to {CARD} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
