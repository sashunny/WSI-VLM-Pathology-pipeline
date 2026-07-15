"""
Run the trained patch classifier over all WSI tissue patches
and stitch patch-level predictions back into a slide-level heatmap. This is the
standard deliverable in WSI deep learning: pathologist-facing visual overlay
highlighting regions of interest.

Also runs Grad-CAM on a couple of sample patches as a lightweight, weakly-supervised
localization method (useful when don't have pixel-level segmentation masks, a very
common constraint in real pathology datasets).

Outputs: outputs/slide_heatmap.png, outputs/gradcam_examples/*.png
"""
import os
import glob
import re
import json
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
PATCH_DIR = os.path.join(OUT_DIR, "patches")
GRADCAM_DIR = os.path.join(OUT_DIR, "gradcam_examples")
os.makedirs(GRADCAM_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 224


def load_model():
    ckpt = torch.load(os.path.join(OUT_DIR, "classifier.pt"), map_location=DEVICE)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model, ckpt["classes"]


def preprocess(img: Image.Image):
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    return tf(img).unsqueeze(0)


def parse_xy(filename):
    m = re.search(r"_x(\d+)_y(\d+)", filename)
    return int(m.group(1)), int(m.group(2))


def simple_gradcam(model, x, target_layer):
    """Minimal Grad-CAM: hook the last conv block, weight activations by their
    gradient w.r.t. the predicted class, and produce a coarse localization map."""
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations["value"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    logits = model(x)
    pred_class = logits.argmax(1).item()
    model.zero_grad()
    logits[0, pred_class].backward()

    acts = activations["value"][0]      # (C, H, W)
    grads = gradients["value"][0]        # (C, H, W)
    weights = grads.mean(dim=(1, 2))     # (C,)
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    h1.remove()
    h2.remove()
    return cam, pred_class


def main():
    model, class_names = load_model()
    patch_paths = sorted(glob.glob(os.path.join(PATCH_DIR, "*.png")))
    print(f"Running inference on {len(patch_paths)} patches...")

    coords, pred_classes, pred_confs = [], [], []
    for p in patch_paths:
        img = Image.open(p).convert("RGB")
        x = preprocess(img).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        cls = probs.argmax().item()
        coords.append(parse_xy(os.path.basename(p)))
        pred_classes.append(cls)
        pred_confs.append(probs[cls].item())

    coords = np.array(coords)
    pred_classes = np.array(pred_classes)
    pred_confs = np.array(pred_confs)

    # --- Stitch a slide-level heatmap: color = predicted class, alpha = confidence ---
    xs, ys = coords[:, 0], coords[:, 1]
    grid_w = (xs.max() - xs.min()) // TILE_SIZE + 1
    grid_h = (ys.max() - ys.min()) // TILE_SIZE + 1
    heat = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
    conf_map = np.zeros((grid_h, grid_w), dtype=np.float32)

    cmap = plt.get_cmap("tab10")
    for (x, y), cls, conf in zip(coords, pred_classes, pred_confs):
        gx = (x - xs.min()) // TILE_SIZE
        gy = (y - ys.min()) // TILE_SIZE
        color = np.array(cmap(cls % 10)[:3])
        heat[gy, gx] = color
        conf_map[gy, gx] = conf

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(heat)
    ax.set_title("Slide-level prediction heatmap (color = predicted tissue class)")
    ax.axis("off")
    # legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 10)) for i in range(len(class_names))]
    ax.legend(handles, class_names, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "slide_heatmap.png"), dpi=150)
    print(f"Saved {OUT_DIR}/slide_heatmap.png")

    # class distribution summary (persisted for the report generator in step 4)
    unique, counts = np.unique(pred_classes, return_counts=True)
    dist = {class_names[u]: int(c) for u, c in zip(unique, counts)}
    print("\nPatch-level class distribution:")
    for name, c in dist.items():
        print(f"  {name:40s}: {c:4d} ({c/len(pred_classes)*100:.1f}%)")
    with open(os.path.join(OUT_DIR, "class_distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)

    # --- Grad-CAM on a few sample patches ---
    target_layer = model.layer4[-1]
    for p in patch_paths[:4]:
        img = Image.open(p).convert("RGB")
        x = preprocess(img).to(DEVICE)
        cam, pred_class = simple_gradcam(model, x, target_layer)
        img_resized = np.array(img.resize((224, 224))) / 255.0
        heatmap_color = plt.get_cmap("jet")(cam)[:, :, :3]
        overlay = 0.5 * img_resized + 0.5 * heatmap_color
        out_path = os.path.join(GRADCAM_DIR, os.path.basename(p))
        plt.imsave(out_path, np.clip(overlay, 0, 1))
    print(f"Saved Grad-CAM examples to {GRADCAM_DIR}/")


if __name__ == "__main__":
    main()
