
# scripts/build_demo.py
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any
from torchvision import transforms
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import importlib
import sys
import cmcrameri.cm as cmc

from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules["astroos_pipelines.datasets"])
importlib.reload(sys.modules["astroos_pipelines.config.astroos_config"])
importlib.reload(sys.modules['astroos_pipelines.transforms'])

# -----------------------------
# USER CONFIG
# -----------------------------
SEED = 42
NUM_SAMPLES = 100
OUTPUT_DIR = Path("site")
SAMPLES_DIR = OUTPUT_DIR / "assets" / "samples"
PLOTS_DIR = OUTPUT_DIR / "assets" / "plots"
DATA_DIR = OUTPUT_DIR / "data"


def get_dataset():
    """
    Replace this with your real dataset construction.
    Example:
        from astroos_pipelines.datasets import MyDataset
        return MyDataset(...)
    """
    transformCartesian = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        # transforms.CenterCrop(50),
        # CropAroundCentroid(crop_size=(50, 50)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
        # CropAroundCentroid(crop_size=(30, 30)),
        # CropAroundCentroid(crop_size=(20, 20)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
    ])

    dataset_dir = "data"
    dataset_name = "0404-5"

    dataset_cartesian = FITS_Image_Morphometry_Photometry_Dataset(
        dataset_dir=os.path.join(dataset_dir, dataset_name),
        labels_init_file=os.path.join(dataset_dir, dataset_name, "labels.csv"),
        transform=transformCartesian,
        morphometric_transform=None,
    )

    return dataset_cartesian


# -----------------------------
# HELPERS
# -----------------------------
def ensure_dirs() -> None:
    for d in [SAMPLES_DIR, PLOTS_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def to_numpy_image(x: Any) -> np.ndarray:
    """
    Convert PIL / torch tensor / numpy array to HWC uint8 image for saving.
    Supports grayscale or RGB.
    """
    if isinstance(x, Image.Image):
        arr = np.array(x)
    elif torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(f"Unsupported image type: {type(x)}")

    # Handle CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    # Handle single-channel trailing dim
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    arr = np.nan_to_num(arr)

    # Normalize if float-like
    if np.issubdtype(arr.dtype, np.floating):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        arr = (255 * arr).clip(0, 255).astype(np.uint8)
    else:
        if arr.dtype != np.uint8:
            arr = arr.clip(0, 255).astype(np.uint8)

    return arr


import numpy as np
from pathlib import Path
from PIL import Image
import cmcrameri.cm as cmc

def save_image(arr: np.ndarray, path: Path) -> None:
    arr = arr.astype(np.float32)

    # robust normalization (avoids outliers blowing out contrast)
    vmin, vmax = np.percentile(arr, [1, 99])
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-8)

    # apply batlow colormap → RGBA in [0,1]
    rgba = cmc.batlow(arr)

    # convert to uint8 RGB
    rgb = (rgba[..., :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img.save(path)


def extract_sample(sample: Any) -> tuple[Any, Any, dict[str, Any]]:
    """
    Normalize different dataset return types into:
        image, label, meta_dict
    """
    if isinstance(sample, dict):
        image = sample.get("image")
        label = sample.get("label")
        meta = {k: v for k, v in sample.items() if k not in ("image", "label")}
        return image, label, meta

    if isinstance(sample, (tuple, list)):
        if len(sample) == 2:
            image, label, _, _, _ = sample
            return image, label, {}
        if len(sample) >= 3:
            image, label, meta = sample[0], sample[1], sample[2]

            if (label == 0):
                label = "Star-forming"
            elif (label == 1):
                label = "Quiescent"
            else:
                label = f"Unknown({label})"
            if meta is None:
                meta = {}
            if not isinstance(meta, dict):
                meta = {"meta": str(meta)}
            return image, label, meta

    raise TypeError(
        "Unsupported dataset item format. Expected dict or tuple/list."
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


# -----------------------------
# BUILD
# -----------------------------
def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    ensure_dirs()

    dataset = get_dataset()
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty.")

    sample_count = min(NUM_SAMPLES, n)
    chosen_indices = random.sample(range(n), sample_count)

    samples_out: list[dict[str, Any]] = []
    label_counter = Counter()
    widths = []
    heights = []

    for i, ds_idx in enumerate(chosen_indices):
        raw_sample = dataset[ds_idx]
        image, label, meta = extract_sample(raw_sample)

        n_bands = image.shape[0]
        
        sample_dir = SAMPLES_DIR / f"sample_{i:05d}"
        os.makedirs(sample_dir, exist_ok=True)
        
        for b in range(n_bands):
            arr = to_numpy_image(image[b])
            # if (label == 0):
                # label = "Star-forming"
            # elif (label == 1):
                # label = "Quiescent"
            # else:
                # label = f"Unknown({label})"
            print(f"Sample {i} Band {b}: label={label}, meta={meta}, image_shape={arr.shape}")

            filename = f"sample_{i:05d}_band{b}.png"
            image_path = sample_dir / filename
            save_image(arr, image_path)



        # arr = to_numpy_image(image[0])
        # print(f"Sample {i}: label={label}, meta={meta}, image_shape={arr.shape}")

        # h, w = arr.shape[:2]
        # widths.append(w)
        # heights.append(h)

        label_str = str(json_safe(label))

        label_counter[label_str] += 1

        # filename = f"sample_{i:05d}.png"
        # image_path = SAMPLES_DIR / filename
        # save_image(arr, image_path)

        sample_record = {
            "id": i,
            "dataset_index": ds_idx,
            "image_dir": sample_dir.as_posix(),
            "label": label_str,
            # "width": w,
            # "height": h,
            "meta": {}, # json_safe(meta),
        }
        samples_out.append(sample_record)

    # Plot: class counts
    if label_counter:
        labels = list(label_counter.keys())
        counts = [label_counter[k] for k in labels]

        plt.figure(figsize=(10, 5))
        plt.bar(labels, counts)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("Class Counts")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "class_counts.png", dpi=150)
        plt.close()

    # Plot: width/height scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.7)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.title("Image Width vs Height")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "width_height.png", dpi=150)
    plt.close()

    eda = {
        "num_exported_samples": len(samples_out),
        "dataset_length": n,
        "label_counts": dict(label_counter),
        "width": {
            "min": min(widths) if widths else None,
            "max": max(widths) if widths else None,
            "mean": float(np.mean(widths)) if widths else None,
        },
        "height": {
            "min": min(heights) if heights else None,
            "max": max(heights) if heights else None,
            "mean": float(np.mean(heights)) if heights else None,
        },
        "plots": {
            "class_counts": "assets/plots/class_counts.png",
            "width_height": "assets/plots/width_height.png",
        },
    }

    with open(DATA_DIR / "samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_out, f, indent=2)

    with open(DATA_DIR / "eda.json", "w", encoding="utf-8") as f:
        json.dump(eda, f, indent=2)

    print(f"Exported {len(samples_out)} samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
