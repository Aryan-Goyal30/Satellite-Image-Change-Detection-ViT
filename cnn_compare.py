"""
CNN vs Vision Transformer — Why We Chose ViT
Comparison Script: ResNet-50 vs ViT-Base on same satellite image pair

Project: Satellite Image Change Detection using Vision Transformer

"""

import torch
import torch.nn as nn
import timm
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
before_path = os.path.join(BASE_DIR, "images", "before1.png")
after_path  = os.path.join(BASE_DIR, "images", "after1.png")

# ── Transforms ─────────────────────────────────────────────────────────────────
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Models ─────────────────────────────────────────────────────────────────────
print("Loading CNN (ResNet-50)...")
resnet    = models.resnet50(pretrained=True)
cnn_model = nn.Sequential(*list(resnet.children())[:-2])   # output: 7×7×2048
cnn_model.eval()

print("Loading ViT (vit_base_patch16_224)...")
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
vit_model.eval()

# ══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path, transform):
    img     = Image.open(path).convert("RGB")
    resized = img.resize((224, 224))
    tensor  = transform(resized).unsqueeze(0)
    return resized, tensor


def extract_cnn_features(tensor):
    """ResNet-50 → 7×7 spatial feature map (pooled from local conv kernels)."""
    with torch.no_grad():
        return cnn_model(tensor)             # [1, 2048, 7, 7]


def extract_vit_features(tensor):
    """ViT → 14×14 patch tokens (global self-attention, no pooling)."""
    with torch.no_grad():
        features = vit_model.forward_features(tensor)
        return features[:, 1:, :]           # remove [CLS] → 196 tokens


def compute_cnn_change_map(f_before, f_after):
    diff     = torch.abs(f_before - f_after).mean(dim=1)
    diff_map = diff.squeeze().detach().numpy()   # 7×7
    if diff_map.max() != diff_map.min():
        return (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
    return np.zeros_like(diff_map)


def compute_vit_change_map(f_before, f_after):
    diff     = torch.abs(f_before - f_after).mean(dim=2)
    diff_map = diff.reshape(14, 14).detach().numpy()   # 14×14
    if diff_map.max() != diff_map.min():
        return (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
    return np.zeros_like(diff_map)


def create_overlay(base_img_np, heatmap, grid_size):
    """Upscale heatmap and blend as red overlay on the base image."""
    scale          = 224 // grid_size
    heatmap_up     = np.kron(heatmap, np.ones((scale, scale)))
    heat_intensity = (heatmap_up * 255).astype(np.uint8)
    red_mask       = np.zeros_like(base_img_np)
    red_mask[:, :, 0] = heat_intensity
    return (0.7 * base_img_np + 0.6 * red_mask).clip(0, 255).astype(np.uint8)


def compute_percentage_change(norm_map, sensitivity=1.5):
    """Adaptive threshold: only patches > mean + sensitivity×std are counted."""
    threshold = norm_map.mean() + sensitivity * norm_map.std()
    changed   = np.sum(norm_map > threshold)
    return (changed / norm_map.size) * 100


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBefore: {before_path}")
print(f"After : {after_path}\n")

before_raw_v, before_vit = load_image(before_path, vit_transform)
after_raw_v,  after_vit  = load_image(after_path,  vit_transform)

before_raw_c, before_cnn = load_image(before_path, cnn_transform)
_,            after_cnn  = load_image(after_path,  cnn_transform)

before_np = np.array(before_raw_v)
after_np  = np.array(after_raw_v)

# CNN
t0           = time.time()
fb_cnn       = extract_cnn_features(before_cnn)
fa_cnn       = extract_cnn_features(after_cnn)
cnn_time     = time.time() - t0
cnn_map      = compute_cnn_change_map(fb_cnn, fa_cnn)
cnn_overlay  = create_overlay(np.array(before_raw_c), cnn_map, grid_size=7)
cnn_pct      = compute_percentage_change(cnn_map)

# ViT
t1           = time.time()
fb_vit       = extract_vit_features(before_vit)
fa_vit       = extract_vit_features(after_vit)
vit_time     = time.time() - t1
vit_map      = compute_vit_change_map(fb_vit, fa_vit)
vit_overlay  = create_overlay(before_np, vit_map, grid_size=14)
vit_pct      = compute_percentage_change(vit_map)

# Print table
print("=" * 54)
print(f"{'Metric':<28} {'CNN (ResNet-50)':<14} {'ViT'}")
print("=" * 54)
print(f"{'Spatial Map Resolution':<28} {'7×7  (49)':<14} {'14×14 (196)'}")
print(f"{'Spatial Detail':<28} {'Low':<14} {'4× Higher'}")
print(f"{'Context Mechanism':<28} {'Local Conv':<14} {'Global Attn'}")
print(f"{'Pooling Layers':<28} {'5 (compresses)':<14} {'None'}")
print(f"{'% Area Changed':<28} {cnn_pct:<14.2f} {vit_pct:.2f}")
print(f"{'Inference Time (s)':<28} {cnn_time:<14.3f} {vit_time:.3f}")
print("=" * 54)
print()


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  — 4-row comparison figure
# ══════════════════════════════════════════════════════════════════════════════
LIGHT_BG = "#F4F6FB"

fig = plt.figure(figsize=(20, 20), facecolor=LIGHT_BG)
fig.suptitle(
    "Why ViT over CNN?\nCNN (ResNet-50) vs Vision Transformer — Satellite Image Change Detection",
    fontsize=16, fontweight="bold", y=0.99, color="#1A1A2E"
)

gs = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.45, wspace=0.28,
    top=0.94, bottom=0.03, left=0.05, right=0.97
)

BOX_STYLES = [
    dict(boxstyle='round,pad=0.7', facecolor='#E8EEF9', alpha=0.95, edgecolor="#99AACC"),
    dict(boxstyle='round,pad=0.7', facecolor='#FFF8E1', alpha=0.95, edgecolor="#CCAA44"),
    dict(boxstyle='round,pad=0.7', facecolor='#E8F5E9', alpha=0.95, edgecolor="#66AA77"),
    dict(boxstyle='round,pad=0.7', facecolor='#FDE8E8', alpha=0.95, edgecolor="#CC6666"),
]

def light_ax(ax, title, color="#1A1A2E"):
    ax.set_facecolor(LIGHT_BG)
    ax.set_title(title, color=color, fontsize=11, pad=7, fontweight="bold")
    ax.axis("off")

def info_box(ax, text, box_style, fontsize=10):
    ax.axis("off")
    ax.set_facecolor(LIGHT_BG)
    ax.text(0.05, 0.95, text,
            fontsize=fontsize, va='top', family='monospace',
            transform=ax.transAxes, bbox=box_style)

# ── Row 0: Original images + overview box ────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
ax00.imshow(before_raw_v)
light_ax(ax00, "Before Image")

ax01 = fig.add_subplot(gs[0, 1])
ax01.imshow(after_raw_v)
light_ax(ax01, "After Image")

info_box(fig.add_subplot(gs[0, 2]),
    "Comparison Overview\n"
    "────────────────────────────────────\n"
    "Both models use ImageNet pretrained\n"
    "weights on the SAME image pair.\n\n"
    f"CNN (ResNet-50)\n"
    f"  Resolution : 7×7   = 49 locations\n"
    f"  Mechanism  : Local convolutions\n"
    f"  Pooling    : 5 layers (compresses)\n"
    f"  Changed    : {cnn_pct:.2f}%\n"
    f"  Time       : {cnn_time:.3f}s\n\n"
    f"ViT (vit_base_patch16_224)\n"
    f"  Resolution : 14×14 = 196 locations\n"
    f"  Mechanism  : Global self-attention\n"
    f"  Pooling    : None\n"
    f"  Changed    : {vit_pct:.2f}%\n"
    f"  Time       : {vit_time:.3f}s",
    BOX_STYLES[0]
)

# ── Row 1: Heatmaps + explanation ─────────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.imshow(cnn_map, cmap="Reds", interpolation="nearest")
light_ax(ax10, f"CNN Heatmap — 7×7  ({cnn_pct:.2f}% changed)")

ax11 = fig.add_subplot(gs[1, 1])
ax11.imshow(vit_map, cmap="Reds", interpolation="nearest")
light_ax(ax11, f"ViT Heatmap — 14×14  ({vit_pct:.2f}% changed)")

info_box(fig.add_subplot(gs[1, 2]),
    "Heatmap Resolution\n"
    "────────────────────────────────────\n"
    "CNN applies 5 downsampling\n"
    "(pooling) layers.\n\n"
    "7×7 = only 49 cells to cover\n"
    "the entire 224×224 image.\n"
    "Each cell covers 32×32 px.\n\n"
    "ViT has NO pooling.\n"
    "14×14 = 196 cells.\n"
    "Each cell covers 16×16 px.\n\n"
    "→ ViT gives 4× finer spatial\n"
    "  resolution in the change map.\n"
    "→ Smaller regions of change\n"
    "  are visible in ViT output.",
    BOX_STYLES[1]
)

# ── Row 2: Overlays + explanation ─────────────────────────────────────────────
ax20 = fig.add_subplot(gs[2, 0])
ax20.imshow(cnn_overlay)
light_ax(ax20, "CNN Change Overlay")

ax21 = fig.add_subplot(gs[2, 1])
ax21.imshow(vit_overlay)
light_ax(ax21, "ViT Change Overlay")

info_box(fig.add_subplot(gs[2, 2]),
    "Why Overlays Look Different\n"
    "────────────────────────────────────\n"
    "CNN uses local 3×3 conv kernels.\n"
    "It detects texture edges and\n"
    "local brightness changes.\n"
    "May highlight noise or boundaries\n"
    "instead of true semantic change.\n\n"
    "ViT uses self-attention across\n"
    "ALL 196 patches simultaneously.\n"
    "It understands that a flooded\n"
    "region on the left may relate\n"
    "to water patterns on the right.\n\n"
    "→ ViT detects semantically\n"
    "  consistent change regions,\n"
    "  not just local pixel edges.",
    BOX_STYLES[2]
)

# ── Row 3: Bar chart comparison + final verdict ───────────────────────────────
ax30 = fig.add_subplot(gs[3, 0:2])
ax30.set_facecolor(LIGHT_BG)

categories  = ["CNN\n(ResNet-50)", "ViT\n(vit_base_patch16_224)"]
pct_values  = [cnn_pct, vit_pct]
bar_colors  = ["#E57373", "#42A5F5"]
x           = np.array([0, 1])
bars        = ax30.bar(x, pct_values, color=bar_colors, width=0.4,
                       edgecolor="#888", linewidth=0.8)

ax30.set_xticks(x)
ax30.set_xticklabels(categories, fontsize=12, fontweight="bold")
ax30.set_ylabel("% Area Detected as Changed", fontsize=11)
ax30.set_title("Percentage Area Changed — CNN vs ViT", fontsize=12, fontweight="bold", pad=8)
ax30.set_ylim(0, max(pct_values) * 1.5 if max(pct_values) > 0 else 1)
ax30.grid(axis='y', linestyle='--', alpha=0.4)

for bar, val in zip(bars, pct_values):
    ax30.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pct_values) * 0.03,
              f"{val:.2f}%", ha='center', fontsize=13, fontweight="bold")

# Resolution comparison inset
ax30_twin = ax30.twinx()
res_values = [49, 196]
ax30_twin.bar(x + 0.22, res_values, color=["#FFCC80", "#81C784"],
              width=0.18, edgecolor="#888", linewidth=0.8, alpha=0.7, label="Patch Count")
ax30_twin.set_ylabel("Number of Spatial Patches", fontsize=10, color="#555")
ax30_twin.tick_params(axis='y', labelcolor="#555")
ax30_twin.legend(loc="upper right", fontsize=9)

info_box(fig.add_subplot(gs[3, 2]),
    "Final Verdict\n"
    "────────────────────────────────────\n"
    "CNN is good for local texture\n"
    "detection but loses spatial\n"
    "detail due to pooling.\n\n"
    "ViT preserves full spatial\n"
    "resolution (14×14 vs 7×7)\n"
    "and uses global self-attention\n"
    "to detect semantically\n"
    "meaningful change regions.\n\n"
    "For satellite imagery where\n"
    "change spans large areas with\n"
    "long-range spatial context,\n"
    "ViT is the better choice.\n\n"
    "→ Our system uses ViT.",
    BOX_STYLES[3]
)

# ── Save ──
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
out_path = os.path.join(BASE_DIR, "outputs", "output_cnn_vs_vit.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=LIGHT_BG)
print(f"Output saved → {out_path}")
plt.show()