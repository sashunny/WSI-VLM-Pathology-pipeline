"""
helpers for WSI tiling and tissue detection.

The core idea: a WSI is too large to process whole, and most of it is empty glass.
So we (1) pick a working magnification level, (2) find tissue regions cheaply on a
downsampled thumbnail, (3) only extract/patch full-res tiles where tissue is present.
"""
import numpy as np
import cv2
import openslide
from PIL import Image


def get_tissue_mask(thumbnail: Image.Image, sat_thresh: int = 20) -> np.ndarray:
    """
    Cheap tissue detector: convert to HSV, threshold on the saturation channel.
    Glass background is low-saturation (near-white/gray); tissue is more saturated
    due to staining (H&E, IHC, etc). This is the standard first-pass filter used
    across most WSI pipelines (CLAM, histolab) before any deep learning happens.
    """
    img = np.array(thumbnail.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, sat_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # remove small speckle noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask  # 255 = tissue, 0 = background


def tissue_fraction(mask_patch: np.ndarray) -> float:
    return float((mask_patch > 0).mean())


def iter_tile_coords(slide: openslide.OpenSlide, level: int, tile_size: int, stride: int = None):
    """
    Yield (x, y) top-left coordinates (in level-0 / full-res pixel space, as OpenSlide
    expects) for a grid of tiles at the given pyramid level.
    """
    stride = stride or tile_size
    level_w, level_h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    for y in range(0, level_h - tile_size + 1, stride):
        for x in range(0, level_w - tile_size + 1, stride):
            # convert level coords -> level-0 coords required by read_region
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            yield x, y, x0, y0


def read_tile(slide: openslide.OpenSlide, level: int, x0: int, y0: int, tile_size: int) -> Image.Image:
    tile = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
    return tile
