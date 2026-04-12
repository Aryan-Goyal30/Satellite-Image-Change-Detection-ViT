"""
Satellite Image Change Detection — Vision Transformer (ViT)
Main System: Change Detection + Multi-Class Map + Per-Class Bar Chart

Project: Satellite Image Change Detection using Vision Transformer
Author: Aryan Goyal(2427030332) and Aryan Tyagi(2427030344)
"""

import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(BASE_DIR, "images", "before1.png")
after_path  = os.path.join(BASE_DIR, "images", "after1.png")

# ── Model ──────────────────────────────────────────────────────────────────────
print("Loading ViT model (vit_base_patch16_224)...")
model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()

# ── Transform ──────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# ══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path):
    img     = Image.open(path).convert("RGB")
    resized = img.resize((224, 224))
    tensor  = transform(resized).unsqueeze(0)
    return resized, tensor


def extract_features(tensor):
    """Extract 14×14 patch-level embeddings from ViT (196 tokens)."""
    with torch.no_grad():
        features = model.forward_features(tensor)
        features = features[:, 1:, :]   # Remove [CLS] token → 196 patch tokens
    return features


def compute_change_map(f_before, f_after):
    """
    Compute per-patch L1 distance and normalise to [0, 1].
    Output: 14×14 numpy array.
    """
    diff     = torch.abs(f_before - f_after).mean(dim=2)
    diff_map = diff.reshape(14, 14).detach().numpy()
    if diff_map.max() != diff_map.min():
        return (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
    return np.zeros_like(diff_map)


def create_overlay(base_img_np, heatmap_14):
    """Upscale 14×14 heatmap to 224×224 and blend as red overlay."""
    heatmap_up    = np.kron(heatmap_14, np.ones((16, 16)))
    heat_intensity = (heatmap_up * 255).astype(np.uint8)
    red_mask       = np.zeros_like(base_img_np)
    red_mask[:, :, 0] = heat_intensity
    return (0.7 * base_img_np + 0.6 * red_mask).clip(0, 255).astype(np.uint8)


def compute_percentage_change(norm_map, sensitivity=1.5):
    """
    Adaptive threshold = mean + sensitivity × std.
    Only patches significantly above average are counted as changed.

    sensitivity=1.5 → stricter (fewer false positives)
    sensitivity=1.0 → more sensitive (catches subtler changes)
    """
    threshold      = norm_map.mean() + sensitivity * norm_map.std()
    changed_patches = np.sum(norm_map > threshold)
    total_patches   = norm_map.size   # 196 (14×14)
    return (changed_patches / total_patches) * 100, changed_patches, total_patches


# ── Multi-Class Detection ──────────────────────────────────────────────────────
CLASS_INFO = {
    "Flood":             {"rgb": [0,   0,   255], "hex": "#0055FF"},
    "Vegetation Loss":   {"rgb": [0,   200,   0], "hex": "#00C800"},
    "New Construction":  {"rgb": [255, 220,   0], "hex": "#FFDC00"},
    "Fire / Burned":     {"rgb": [255,  50,   0], "hex": "#FF3200"},
    "Snow / Ice Change": {"rgb": [220, 220, 220], "hex": "#DCDCDC"},
}

def multi_class_detection(before_np, after_np):
    """
    Rule-based spectral classification on RGB pixel values.
    Returns coloured class map (224×224×3) and per-class percentages.
    """
    total = before_np.shape[0] * before_np.shape[1]
    delta = after_np.astype(np.int16) - before_np.astype(np.int16)

    masks = {
        "Flood": (
            (after_np[:, :, 2] > after_np[:, :, 1]) &
            (after_np[:, :, 2] > after_np[:, :, 0]) &
            (delta[:, :, 2] > 20)
        ),
        "Vegetation Loss": (
            (before_np[:, :, 1] > 120) &
            (after_np[:, :, 1] < 80)
        ),
        "New Construction": (delta.mean(axis=2) > 25),
        "Fire / Burned": (
            (before_np.mean(axis=2) - after_np.mean(axis=2)) > 40
        ),
        "Snow / Ice Change": (
            ((after_np.mean(axis=2)  > 200) & (before_np.mean(axis=2) < 180)) |
            ((before_np.mean(axis=2) > 200) & (after_np.mean(axis=2)  < 180))
        ),
    }

    percentages = {
        label: (np.sum(mask) / total) * 100
        for label, mask in masks.items()
    }

    class_map = np.zeros((224, 224, 3), dtype=np.uint8)
    for label, mask in masks.items():
        class_map[mask] = CLASS_INFO[label]["rgb"]

    return class_map, percentages


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBefore: {before_path}")
print(f"After : {after_path}\n")

before_raw, before_tensor = load_image(before_path)
after_raw,  after_tensor  = load_image(after_path)

before_np = np.array(before_raw)
after_np  = np.array(after_raw)

# Feature extraction & change map
f_before  = extract_features(before_tensor)
f_after   = extract_features(after_tensor)
norm_map  = compute_change_map(f_before, f_after)

# Overlays
overlay       = create_overlay(before_np, norm_map)
class_map, class_pcts = multi_class_detection(before_np, after_np)
multi_overlay = (0.6 * before_np + 0.4 * class_map).clip(0, 255).astype(np.uint8)

# Percentage change
overall_pct, changed_patches, total_patches = compute_percentage_change(norm_map)

# Print summary
print("─" * 40)
print(f"  Changed Patches : {changed_patches} / {total_patches}")
print(f"  Overall Change  : {overall_pct:.2f}%")
print("  Per-Class Breakdown:")
for label, pct in class_pcts.items():
    bar = "█" * int(pct * 2)
    print(f"    {label:<22}: {pct:.2f}%  {bar}")
print("─" * 40)
print("  (Pretrained weights — conservative % is expected and honest.)\n")


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  — dark-theme 6-panel figure
# ══════════════════════════════════════════════════════════════════════════════
BG = "#0D1117"

fig = plt.figure(figsize=(20, 11), facecolor=BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.22)

def dark_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color="white", fontsize=11, pad=7, fontweight="bold")
    ax.axis("off")

# ── Panel 1: Before ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(before_raw)
dark_ax(ax1, "Before Image")

# ── Panel 2: After ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(after_raw)
dark_ax(ax2, "After Image")

# ── Panel 3: ViT Heatmap ──
ax3 = fig.add_subplot(gs[0, 2])
im = ax3.imshow(norm_map, cmap="hot", interpolation="nearest")
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")
dark_ax(ax3, "ViT Change Heatmap (14×14 Patch Map)")

# ── Panel 4: ViT Overlay ──
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(overlay)
dark_ax(ax4, "ViT Change Overlay  (Red = Changed Area)")

# ── Panel 5: Multi-Class Map ──
ax5 = fig.add_subplot(gs[1, 1])
ax5.imshow(multi_overlay)
patches = [
    mpatches.Patch(color=info["hex"], label=label)
    for label, info in CLASS_INFO.items()
]
ax5.legend(handles=patches, loc="lower left", fontsize=8,
           facecolor="#1C2333", labelcolor="white", edgecolor="#444", framealpha=0.9)
dark_ax(ax5, "Multi-Class Change Map")

# ── Panel 6: Per-Class Horizontal Bar Chart ──
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor("#111827")

labels = list(class_pcts.keys())
values = list(class_pcts.values())
colors = [CLASS_INFO[l]["hex"] for l in labels]
y_pos  = np.arange(len(labels)) * 1.35

bars = ax6.barh(y_pos, values, color=colors, height=0.7, edgecolor="#2a2a2a", linewidth=0.5)

ax6.set_yticks(y_pos)
ax6.set_yticklabels(labels, color="white", fontsize=10)
ax6.tick_params(axis='x', colors='white', labelsize=9)
ax6.set_xlabel("% of Image Area", color="#AAAAAA", fontsize=10)

max_val = max(values) if max(values) > 0 else 1
ax6.set_xlim(0, max_val * 1.4)

for bar, val in zip(bars, values):
    ax6.text(
        bar.get_width() + max_val * 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center", color="white", fontsize=9
    )

ax6.set_title(
    f"Per-Class Change  |  Overall: {overall_pct:.2f}%",
    color="white", fontsize=11, pad=7, fontweight="bold"
)
for spine in ax6.spines.values():
    spine.set_edgecolor("#444")
ax6.grid(axis='x', linestyle='--', alpha=0.2, color="white")
ax6.set_facecolor("#111827")

# ── Main title ──
fig.suptitle(
    "Satellite Image Change Detection  —  Vision Transformer (ViT)",
    color="white", fontsize=15, fontweight="bold", y=0.98
)

# ── Save ──
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
out_path = os.path.join(BASE_DIR, "outputs", "output_vit_result.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Output saved → {out_path}")
plt.show()