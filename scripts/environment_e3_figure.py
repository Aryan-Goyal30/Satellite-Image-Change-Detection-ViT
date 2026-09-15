"""E3 qualitative comparison: the same test samples under E2 and E3.

Seven panels per sample - BEFORE, AFTER, GT at 30 m support, E2 at 10 m,
E3 at 10 m, E3 at 30 m, and a TP/FP/FN error view at 30 m support, which is
where the hypothesis lives.

Cases are selected by the Phase 4B-3 error taxonomy and frozen dataset
metadata, never by E3's own performance.

Usage:  python scripts/environment_e3_figure.py
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                                              # noqa: E402
from src.domains.environment import support as sp                   # noqa: E402
from src.domains.environment.data import sentinel2 as s2            # noqa: E402
from src.domains.environment.data.loader import (                   # noqa: E402
    EnvironmentChangeDataset, load_band_statistics)
from src.domains.environment.model import BAND_SETS, build_model    # noqa: E402

DATASET_DIR = os.path.join(config.DATA_DIR, "environment", "dataset_v24")
STATS_PATH = os.path.join(config.DATA_DIR, "environment",
                          "characterization_v24_train_bands.json")
OUT_DIR = os.path.join(config.OUTPUTS, "environment_e3")
MODELS = {
    "E2": (os.path.join(config.OUTPUTS, "environment_baseline"),
           "environment_sixband_best.pt"),
    "E3": (OUT_DIR, "environment_e3_best.pt"),
}
FIXED_MAX, FIXED_GAMMA = 0.30, 0.6


@torch.no_grad()
def infer(code, device):
    directory, name = MODELS[code]
    payload = torch.load(os.path.join(directory, name), map_location=device,
                         weights_only=False)
    bands = tuple(payload.get("bands") or BAND_SETS["six_band"])
    stats = load_band_statistics(STATS_PATH, bands)
    model = build_model(bands=bands, encoder=payload.get("encoder", "resnet34"),
                        pretrained=False)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    dataset = EnvironmentChangeDataset(DATASET_DIR, "test", stats, bands=bands)
    amp = torch.bfloat16 if (device.type == "cuda"
                             and torch.cuda.is_bf16_supported()) else torch.float16
    fine, coarse = {}, {}
    for batch in DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0):
        with torch.autocast("cuda", dtype=amp, enabled=device.type == "cuda"):
            logits = model(batch["before"].to(device), batch["after"].to(device))
        logits = logits.float()
        p10 = torch.sigmoid(logits).cpu().numpy()
        p30 = torch.sigmoid(sp.pool_mean_torch(logits)).cpu().numpy()
        for i, sid in enumerate(batch["sample_id"]):
            fine[sid] = p10[i, 0]
            coarse[sid] = p30[i, 0]
    return fine, coarse


def composite(reflectance):
    idx = [s2.BAND_INDEX[b] for b in ("B04", "B03", "B02")]
    scaled = np.stack([reflectance[:, :, i] for i in idx], -1) / FIXED_MAX
    return np.clip(scaled, 0, 1) ** FIXED_GAMMA


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(OUT_DIR, "test_evaluation.json"), encoding="utf-8") as fh:
        thresholds = json.load(fh)["thresholds"]
    with open(os.path.join(DATASET_DIR, "manifest.json"), encoding="utf-8") as fh:
        records = {r["sample_id"]: r for r in json.load(fh)["samples"]
                   if r["split"] == "test"}
    with open(os.path.join(config.OUTPUTS, "environment_failure_analysis",
                           "per_sample_errors.json"), encoding="utf-8") as fh:
        table = {r["sample_id"]: r for r in json.load(fh)}

    probs = {}
    for code in MODELS:
        probs[code] = infer(code, device)

    def pick(predicate, key, reverse=True):
        members = [r for r in table.values() if predicate(r)]
        if not members:
            return None
        return sorted(members, key=lambda r: (key(r), r["sample_id"]),
                      reverse=reverse)[0]["sample_id"]

    wanted = [
        ("strong E2 case", pick(lambda r: r["primary_category"] == "STRONG_DETECTION",
                                lambda r: r["iou"])),
        ("fragmentation case", pick(lambda r: r["primary_category"] == "FRAGMENTATION_MISS",
                                    lambda r: r["gt_components"])),
        ("large event", pick(lambda r: r["sample_type"] == "positive"
                             and r["event_size"] in ("large", "very_large"),
                             lambda r: r["true_px"])),
        ("small event", pick(lambda r: r["sample_type"] == "positive",
                             lambda r: -r["true_px"])),
        ("complete miss", pick(lambda r: r["primary_category"] == "COMPLETE_MISS",
                               lambda r: r["true_px"])),
        ("false positive", pick(lambda r: r["sample_type"] == "negative",
                                lambda r: r["fp"])),
        ("Amazon", pick(lambda r: r["region"] == "Amazon" and r["sample_type"] == "positive",
                        lambda r: r["true_px"])),
        ("SE Asia", pick(lambda r: r["region"] == "SE Asia" and r["sample_type"] == "positive",
                         lambda r: r["true_px"])),
    ]
    cases, used = [], set()
    for caption, sid in wanted:
        if sid and sid not in used:
            used.add(sid)
            cases.append((sid, caption))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    binary = ListedColormap(["#111111", "#ff2d2d"])
    errors = ListedColormap(["#111111", "#2ecc71", "#3498db", "#f1c40f"])
    columns = 7
    fig, axes = plt.subplots(len(cases), columns,
                             figsize=(2.9 * columns, 3.1 * len(cases)))
    if len(cases) == 1:
        axes = np.array([axes])

    for row, (sid, caption) in enumerate(cases):
        record = records[sid]
        before = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_before.npy")),
                                   record["before_offset_applied"], record["before_baseline"])
        after = s2.to_reflectance(np.load(os.path.join(DATASET_DIR, f"{sid}_after.npy")),
                                  record["after_offset_applied"], record["after_baseline"])
        gt10 = np.load(os.path.join(DATASET_DIR, f"{sid}_label.npy")) > 0.5
        gt30 = sp.label_30m(gt10) > 0.5
        e2_10 = probs["E2"][0][sid] >= thresholds["E2_10m"]
        e3_10 = probs["E3"][0][sid] >= thresholds["E3_10m"]
        e3_30 = probs["E3"][1][sid] >= thresholds["E3_30m"]

        err = np.zeros(gt30.shape, np.uint8)
        err[np.logical_and(e3_30, gt30)] = 1
        err[np.logical_and(e3_30, ~gt30)] = 2
        err[np.logical_and(~e3_30, gt30)] = 3
        inter = np.logical_and(e3_30, gt30).sum()
        union = np.logical_or(e3_30, gt30).sum()
        iou30 = float(inter / union) if union else 1.0

        panels = [
            (composite(before), f"BEFORE {record['before_datetime'][:10]}", None, 1),
            (composite(after), f"AFTER {record['after_datetime'][:10]}", None, 1),
            (gt30, f"GT @30m  {100*gt30.mean():.2f}%", binary, 1),
            (e2_10, f"E2 @10m  th {thresholds['E2_10m']:.3f}", binary, 1),
            (e3_10, f"E3 @10m  th {thresholds['E3_10m']:.3f}", binary, 1),
            (e3_30, f"E3 @30m  th {thresholds['E3_30m']:.3f}", binary, 1),
            (err, f"E3 @30m TP/FP/FN  IoU {iou30:.3f}", errors, 3),
        ]
        for col, (image, title, cmap, vmax) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image) if cmap is None else ax.imshow(image, cmap=cmap,
                                                            vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{sid}\n{record['mgrs_tile']}\n{caption}",
                                fontsize=7, rotation=0, ha="right", va="center",
                                labelpad=68)

    fig.legend(handles=[Patch(color="#2ecc71", label="TP"),
                        Patch(color="#3498db", label="FP"),
                        Patch(color="#f1c40f", label="FN")],
               loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Phase 4B-4 E3 - native 30 m supervision vs the E2 baseline, frozen TEST split\n"
                 "fixed reflectance scale 0-0.30 gamma 0.6; thresholds selected on validation only; "
                 "cases chosen by the 4B-3 taxonomy", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    path = os.path.join(OUT_DIR, "qualitative_comparison.png")
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
