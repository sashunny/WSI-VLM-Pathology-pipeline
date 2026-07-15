"""
Load a WSI, detect tissue vs. background, tile it, and save patches.
Inputs:  data/CMU-1.svs
Outputs: outputs/thumbnail_with_tissue_mask.png
         outputs/tile_grid_overlay.png
         outputs/patches/*.png
"""
import os
import openslide
import numpy as np
import cv2
from PIL import Image, ImageDraw
from utils import get_tissue_mask, tissue_fraction, iter_tile_coords, read_tile

SLIDE_PATH = "data/CMU-1.svs"
OUT_DIR = "outputs"
PATCH_DIR = os.path.join(OUT_DIR, "patches")
TILE_SIZE = 224           # matches standard CNN input size
TISSUE_MIN_FRACTION = 0.5  # keep tile only if >=50% tissue
MAX_PATCHES = 300          # cap for a quick demo run

os.makedirs(PATCH_DIR, exist_ok=True)


def main():
    slide = openslide.OpenSlide(SLIDE_PATH)
    print(f"Slide dimensions (level 0): {slide.dimensions}")
    print(f"Pyramid levels available: {slide.level_count}")
    print(f"Downsample factors per level: {slide.level_downsamples}")

    # 1. Thumbnail + tissue mask (cheap, low-res pass)
    thumb = slide.get_thumbnail((1024, 1024))
    mask = get_tissue_mask(thumb)
    overlay = np.array(thumb.convert("RGB")).copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(os.path.join(OUT_DIR, "thumbnail_with_tissue_mask.png"))
    print("Saved thumbnail_with_tissue_mask.png")

    # 2. Pick a working pyramid level for tiling (roughly 20x-equivalent if available)
    level = min(1, slide.level_count - 1)
    print(f"Tiling at pyramid level {level} (downsample={slide.level_downsamples[level]:.1f}x)")

    # mask is defined on the thumbnail; build a resize map from tile-grid coords -> mask coords
    mask_h, mask_w = mask.shape
    level_w, level_h = slide.level_dimensions[level]

    saved = 0
    draw_thumb = thumb.convert("RGB").copy()
    draw = ImageDraw.Draw(draw_thumb)

    for x, y, x0, y0 in iter_tile_coords(slide, level, TILE_SIZE, stride=TILE_SIZE):
        if saved >= MAX_PATCHES:
            break
        # map tile location into mask's coordinate space to check tissue fraction cheaply
        mx0 = int(x / level_w * mask_w)
        my0 = int(y / level_h * mask_h)
        mx1 = max(mx0 + 1, int((x + TILE_SIZE) / level_w * mask_w))
        my1 = max(my0 + 1, int((y + TILE_SIZE) / level_h * mask_h))
        mask_patch = mask[my0:my1, mx0:mx1]
        if mask_patch.size == 0 or tissue_fraction(mask_patch) < TISSUE_MIN_FRACTION:
            continue

        tile = read_tile(slide, level, x0, y0, TILE_SIZE)
        tile.save(os.path.join(PATCH_DIR, f"patch_{saved:04d}_x{x}_y{y}.png"))

        # draw kept tile location on thumbnail for a sanity-check visualization
        draw.rectangle([mx0, my0, mx1, my1], outline=(255, 0, 0), width=1)
        saved += 1

    draw_thumb.save(os.path.join(OUT_DIR, "tile_grid_overlay.png"))
    print(f"Saved {saved} tissue patches to {PATCH_DIR}")
    print("Saved tile_grid_overlay.png (red boxes = kept tissue tiles)")


if __name__ == "__main__":
    main()
