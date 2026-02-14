import torch
import timm
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


model = timm.create_model("vit_base_patch16_224", pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    resized = img.resize((224, 224))
    tensor = transform(resized).unsqueeze(0)
    return resized, tensor

def extract_features(img_tensor):
    with torch.no_grad():
        features = model.forward_features(img_tensor)
        features = features[:, 1:, :]
    return features


def compute_change_map(f_before, f_after):
    diff = torch.abs(f_before - f_after).mean(dim=2)
    diff_map = diff.reshape(14, 14).detach().numpy()
    if diff_map.max() != diff_map.min():
        norm_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())
    else:
        norm_map = np.zeros_like(diff_map)
    return norm_map


def create_overlay(base_img_np, heatmap_14):
    heatmap_up = np.kron(heatmap_14, np.ones((16, 16)))
    heat_intensity = (heatmap_up * 255).astype(np.uint8)
    red_mask = np.zeros_like(base_img_np)
    red_mask[:, :, 0] = heat_intensity
    overlay = (0.7 * base_img_np + 0.6 * red_mask).clip(0, 255).astype(np.uint8)
    return overlay

def compute_percentage_change(norm_map):
    threshold = norm_map.mean() + norm_map.std()
    changed_pixels = np.sum(norm_map > threshold)
    total_pixels = norm_map.size
    return (changed_pixels / total_pixels) * 100


def multi_class_detection(before_np, after_np):
    delta = after_np.astype(np.int16) - before_np.astype(np.int16)
    flood_mask = (
        (after_np[:, :, 2] > after_np[:, :, 1]) &
        (after_np[:, :, 2] > after_np[:, :, 0]) &
        (delta[:, :, 2] > 20)
    )

    veg_loss_mask = (
        (before_np[:, :, 1] > 120) &
        (after_np[:, :, 1] < 80)
    )

    building_mask = (delta.mean(axis=2) > 25)

    fire_mask = (
        (before_np.mean(axis=2) - after_np.mean(axis=2)) > 40
    )

    snow_mask = (
        ((after_np.mean(axis=2) > 200) & (before_np.mean(axis=2) < 180)) |
        ((before_np.mean(axis=2) > 200) & (after_np.mean(axis=2) < 180))
    )

    multi_class = np.zeros((224, 224, 3), dtype=np.uint8)
    multi_class[flood_mask] = [0, 0, 255]       # Blue
    multi_class[veg_loss_mask] = [0, 255, 0]    # Green
    multi_class[building_mask] = [255, 255, 0]  # Yellow
    multi_class[fire_mask] = [255, 0, 0]        # Red
    multi_class[snow_mask] = [255, 255, 255]    # White

    return multi_class


# MAIN EXECUTION

# Correct image folder path
before_path = os.path.join("images", "before.png")
after_path = os.path.join("images", "after.png")

before_raw, before_tensor = load_image(before_path)
after_raw, after_tensor = load_image(after_path)

before_np = np.array(before_raw)
after_np = np.array(after_raw)

# Feature Extraction
f_before = extract_features(before_tensor)
f_after = extract_features(after_tensor)

# Compute change map
norm_map = compute_change_map(f_before, f_after)

# Create overlay
overlay = create_overlay(before_np, norm_map)

# Multi-class detection
multi_class = multi_class_detection(before_np, after_np)
multi_overlay = (0.6 * before_np + 0.4 * multi_class).clip(0, 255).astype(np.uint8)

# Percentage calculation
percentage_change = compute_percentage_change(norm_map)

print(f"\nPercentage Area Changed: {percentage_change:.2f}%\n")


# Visualization

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.imshow(before_raw)
plt.title("Before Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(after_raw)
plt.title("After Image")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(norm_map, cmap="Reds")
plt.title("Heatmap (14x14 Patch Map)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(overlay)
plt.title("ViT Change Overlay")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(multi_overlay)
plt.title("Multi-Class Change Map")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.text(0.2, 0.5, f"Percentage Change:\n{percentage_change:.2f}%", fontsize=18)
plt.axis("off")

plt.tight_layout()
plt.show()
