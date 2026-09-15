"""Compare the Phase 4B-2 spectral ablation: E0 RGB, E1 RGB+NIR, E2 six-band.

Reads the three finished experiments and produces the comparison table, the
deltas, and a qualitative figure in which all three models are shown on the
SAME test samples.

Sample selection is deliberately performance-blind. The samples are chosen from
frozen DATASET METADATA only - positive fraction, region, sample type - never
from any model's output. Choosing by IoU would let the figure flatter whichever
model happened to do well on the samples it selected.

Usage:  python scripts/compare_environment_ablation.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import build_model               # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
ABLATION = os.path.join(config.OUTPUTS, "environment_ablation")
EXPERIMENTS = [
    ("E0", "RGB", os.path.join(ABLATION, "E0_rgb")),
    ("E1", "RGB+NIR", os.path.join(ABLATION, "E1_rgb_nir")),
    ("E2", "6-band", os.path.join(config.OUTPUTS, "environment_baseline")),
]
FIXED_SCALE_MAX, FIXED_SCALE_GAMMA = 0.30, 0.6


def load_experiment(directory):
    with open(os.path.join(directory, "experiment_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(directory, "evaluation.json"), encoding="utf-8") as fh:
        evaluation = json.load(fh)
    return manifest, evaluation


def checkpoint_path(directory):
    for name in ("environment_best.pt", "environment_sixband_best.pt"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(directory)


def choose_samples():
    """A fixed, metadata-driven subset of the test split.

    Chosen before any model is run, from the frozen manifest alone.
    """
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = [r for r in json.load(fh)["samples"] if r["split"] == "test"]
    positives = sorted((r for r in records if r["sample_type"] == "positive"),
                       key=lambda r: r["positive_fraction"])
    negatives = sorted((r for r in records if r["sample_type"] == "negative"),
                       key=lambda r: r["sample_id"])
    asia = [r for r in positives if r["tmf_tile"] in ("N0_E110", "N10_E100")]
    amazon = [r for r in positives if r["tmf_tile"] not in ("N0_E110", "N10_E100")]

    picks, used = [], set()

    def take(record, caption):
        if record is not None and record["sample_id"] not in used:
            used.add(record["sample_id"])
            picks.append((record, caption))

    if positives:
        take(positives[-1], "largest event (highest positive fraction)")
        take(positives[len(positives) // 2], "median event")
        take(positives[0], "smallest event (lowest positive fraction)")
    take(negatives[0] if negatives else None, "negative (first by id)")
    take(asia[-1] if asia else None, "SE Asia, largest positive")
    take(amazon[len(amazon) // 3] if amazon else None, "Amazon, lower-quartile event")
    return picks


@torch.no_grad()
def predict(directory, records, device):
    """Probability maps from one experiment for the chosen samples."""
    payload = torch.load(checkpoint_path(directory), map_location=device,
                         weights_only=False)
    bands = tuple(payload.get("bands") or s2.BANDS)
    stats = load_band_statistics(STATS_PATH, bands)
    model = build_model(bands=bands, encoder=payload.get("encoder", "resnet34"),
                        pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    dataset = EnvironmentChangeDataset(DATASET_DIR, "test", stats, bands=bands)
    index = {r["sample_id"]: i for i, r in enumerate(dataset.records)}
    out = {}
    for record in records:
        item = dataset[index[record["sample_id"]]]
        before = item["before"][None].to(device)
        after = item["after"][None].to(device)
        logits = model(before, after)
        out[record["sample_id"]] = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    return out, bands


def composite(reflectance):
    rgb = [s2.BAND_INDEX[b] for b in ("B04", "B03", "B02")]
    scaled = np.stack([reflectance[:, :, i] for i in rgb], -1) / FIXED_SCALE_MAX
    return np.clip(scaled, 0, 1) ** FIXED_SCALE_GAMMA


def render(picks, predictions, thresholds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#111111", "#ff2d2d"])
    columns = 4 + len(EXPERIMENTS)
    fig, axes = plt.subplots(len(picks), columns,
                             figsize=(3.0 * columns, 3.2 * len(picks)))
    if len(picks) == 1:
        axes = np.array([axes])

    for row, (record, caption) in enumerate(picks):
        sid = record["sample_id"]
        before = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_before.npy")),
                                   record["before_offset_applied"], record["before_baseline"])
        after = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_after.npy")),
                                  record["after_offset_applied"], record["after_baseline"])
        truth = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5

        axes[row, 0].imshow(composite(before))
        axes[row, 0].set_title(f"BEFORE {record['before_datetime'][:10]}", fontsize=8)
        axes[row, 1].imshow(composite(after))
        axes[row, 1].set_title(f"AFTER {record['after_datetime'][:10]}", fontsize=8)
        axes[row, 2].imshow(truth, cmap=cmap, vmin=0, vmax=1)
        axes[row, 2].set_title(f"TMF label {100*truth.mean():.2f}%", fontsize=8)
        axes[row, 3].axis("off")

        for offset, (code, label, _) in enumerate(EXPERIMENTS):
            probability = predictions[code][sid]
            prediction = probability >= thresholds[code]
            inter = np.logical_and(prediction, truth).sum()
            union = np.logical_or(prediction, truth).sum()
            iou = float(inter / union) if union else 1.0
            ax = axes[row, 4 + offset]
            ax.imshow(prediction, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{code} {label}\n@{thresholds[code]:.3f}  IoU {iou:.3f}",
                         fontsize=8)

        axes[row, 0].set_ylabel(f"{sid}\n{record['mgrs_tile']}\n{caption}",
                                fontsize=7, rotation=0, ha="right", va="center",
                                labelpad=64)
        for col in range(columns):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle("Phase 4B-2 spectral ablation - same frozen TEST samples, three band sets\n"
                 "samples chosen from dataset metadata only, never from model output; "
                 "each model uses its own validation-selected threshold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded, table = {}, []
    for code, label, directory in EXPERIMENTS:
        manifest, evaluation = load_experiment(directory)
        loaded[code] = (manifest, evaluation)
        test, region = evaluation["test"]["pixel"], evaluation["test"]["region"]
        table.append({
            "experiment": code, "bands": label,
            "n_bands": len(manifest.get("bands", s2.BANDS)),
            "parameters": manifest["architecture"]["parameters"],
            "stem_scale": manifest["architecture"]["stem_scale"],
            "normalisation_subset_sha256": manifest["normalisation"].get("subset_sha256"),
            "pos_weight": manifest["loss"]["pos_weight"],
            "epochs_run": manifest["epochs_run"], "best_epoch": manifest["best_epoch"],
            "training_seconds": manifest["training_seconds"],
            "val_f1": manifest["best_val_f1"], "threshold": evaluation["threshold"],
            "test_precision": test["precision"], "test_recall": test["recall"],
            "test_f1": test["f1"], "test_iou": test["iou"],
            "test_ap": test["average_precision"],
            "test_pixel_accuracy": test["pixel_accuracy"],
            "tp": test["tp"], "fp": test["fp"], "fn": test["fn"], "tn": test["tn"],
            "region_f1": region["region_f1"],
            "region_precision": region["region_precision"],
            "region_recall": region["region_recall"],
            "predicted_regions": region["predicted_regions"],
            "ground_truth_regions": region["ground_truth_regions"],
            "by_region": {k: {"samples": v["samples"],
                              "positive_samples": v["positive_samples"],
                              "f1": v["pixel"]["f1"], "iou": v["pixel"]["iou"],
                              "ap": v["pixel"]["average_precision"]}
                          for k, v in evaluation["test_by_region"].items()},
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        })

    by_code = {row["experiment"]: row for row in table}
    deltas = {}
    for a, b in (("E1", "E0"), ("E2", "E1"), ("E2", "E0")):
        deltas[f"{a}-{b}"] = {
            metric: round(by_code[a][metric] - by_code[b][metric], 4)
            for metric in ("test_f1", "test_iou", "test_ap", "test_precision",
                           "test_recall", "region_f1", "val_f1")}

    picks = choose_samples()
    predictions, thresholds = {}, {}
    for code, _, directory in EXPERIMENTS:
        maps, bands = predict(directory, [r for r, _ in picks], device)
        predictions[code] = maps
        thresholds[code] = loaded[code][1]["threshold"]
        print(f"  {code}: {len(bands)} bands, threshold {thresholds[code]:.3f}")

    figure = os.path.join(ABLATION, "ablation_qualitative.png")
    os.makedirs(ABLATION, exist_ok=True)
    render(picks, predictions, thresholds, figure)

    summary = {
        "phase": "4B-2 controlled spectral ablation",
        "dataset_version": "v24",
        "controlled": {
            "seed": 20260913, "architecture": "Siamese U-Net, ResNet-34 encoder",
            "optimiser": "AdamW lr 3e-4 wd 1e-4 CosineAnnealingLR(T_max=80)",
            "batch_size": 8, "max_epochs": 80, "early_stopping_patience": 15,
            "loss": "0.5*BCE(pos_weight) + 0.5*(1-soft Dice)",
            "augmentation": "dihedral, paired across before/after/label/invalid",
            "checkpoint_rule": "best validation F1",
            "threshold_rule": "argmax F1 over a 201-point validation sweep",
            "only_variable": "the set of input bands",
        },
        "experiments": table, "deltas": deltas,
        "qualitative_samples": [{"sample_id": r["sample_id"], "caption": c,
                                 "positive_fraction": r["positive_fraction"],
                                 "region": ("SE Asia" if r["tmf_tile"] in
                                            ("N0_E110", "N10_E100") else "Amazon")}
                                for r, c in picks],
        "sample_selection": "frozen dataset metadata only; never model output",
        "figure": os.path.relpath(figure),
    }
    path = os.path.join(ABLATION, "ablation_comparison.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print()
    header = "%-4s %-9s %7s %9s %8s %8s %8s %8s %9s" % (
        "exp", "bands", "params", "val F1", "test F1", "test IoU", "test AP",
        "region F1", "threshold")
    print(header)
    print("-" * len(header))
    for row in table:
        print("%-4s %-9s %7s %9.4f %8.4f %8.4f %8.4f %8.4f %9.3f" % (
            row["experiment"], row["bands"], f"{row['parameters']/1e6:.2f}M",
            row["val_f1"], row["test_f1"], row["test_iou"], row["test_ap"],
            row["region_f1"], row["threshold"]))
    print()
    for key, value in deltas.items():
        print(f"  {key}: " + "  ".join(f"{m} {v:+.4f}" for m, v in value.items()))
    print(f"\n[artifacts] {path}\n            {figure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
