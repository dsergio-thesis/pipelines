
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import os
os.environ["MPLBACKEND"] = "Agg"

import sys
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astropy import units as u
import pandas as pd
import importlib
from io import BytesIO
from pathlib import Path
import hashlib
import requests
import numpy as np
from astropy.io import fits

from astroos_pipelines.dag import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

# importlib.reload(sys.modules['astroos_pipelines.dag'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])


class NodeLSCutoutFetch(Node):
    def __init__(
            self,
            dag_dir=None,
            node_type="catalog_ls_cutout_fetch",
            node_id=None,
            parents=[],
            parameters={},
            label="Fetch LS Cutouts",
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label=label,
            description="Fetch deep coadd cutouts.",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeLSCutoutFetch"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        artifact = self.inputs[0]
        table = artifact.to_table(self.node_id)
        df = table.to_pandas()

        dataset = self.parameters.get("dataset", None) if "dataset" in self.parameters else None
        dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(dataset) if dataset is not None else None

        bands = ["g", "r", "z"]
        stamp_size = int(self.parameters.get("stamp_size", 64))

        cutouts = []
        rows = []

        groups = list(df.groupby("brickname"))

        tasks = [
            (brick_name, group_df.copy(), bands, stamp_size)
            for brick_name, group_df in groups
        ]

        max_workers = int(self.parameters.get("max_workers", 8))
        print(f"Processing {len(tasks)} bricks with {max_workers} workers...")

        all_results = []

        # import sys
        # print("tkinter loaded?", "tkinter" in sys.modules)
        # import sys
        # for name in sys.modules:
            # if "tk" in name.lower():
                # print(name)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(process_brick, task) for task in tasks]

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing bricks"):
                all_results.extend(fut.result())

        for item in tqdm(all_results, desc="Writing dataset"):
            hdu_img = fits.ImageHDU(data=item["data"], name="CUTOUTS")
            hdu_img.header["label"] = item["label"]
            hdu_img.header["ra"] = item["ra"]
            hdu_img.header["dec"] = item["dec"]
            hdu_img.header["objectId"] = item["object_id"]

            if dataset is not None:
                if dataset.contains(item["object_id"]):
                    dataset.update(item["object_id"], hdu_img)
                else:
                    dataset.append(hdu_img)



def download_fits_memory(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    bio = BytesIO(r.content)

    with fits.open(bio, memmap=False) as hdul:
        # hdul.info()

        for hdu in hdul:
            if hdu.data is None:
                continue

            data = np.asarray(hdu.data, dtype=np.float32)

            if data.ndim == 2:
                return data

            if data.ndim == 3:
                return data[0]

    raise ValueError(f"No image found in {url}")


def download_fits_cached(
    url,
    cache_dir="data/ls_cache",
    overwrite=False,
):
    """
    Download FITS once, cache compressed bytes to disk,
    return decompressed numpy image.

    Cache key is md5(url).
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = hashlib.md5(url.encode()).hexdigest() + ".fits.fz"
    cache_path = cache_dir / filename

    # download if missing
    if overwrite or not cache_path.exists():
        # print(f"Downloading {url}")

        r = requests.get(url, timeout=120)
        r.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(r.content)

    # else:
        # print(f"Cache hit: {cache_path}")

    # read cached compressed FITS
    with fits.open(cache_path, memmap=False) as hdul:

        for hdu in hdul:
            if hdu.data is None:
                continue

            data = np.asarray(hdu.data, dtype=np.float32)

            if data.ndim == 2:
                return data

            if data.ndim == 3:
                return data[0]

    raise ValueError(f"No image found in {url}")



def fetch_brick_band(brick_name, band):
    bri = brick_name[:3]
    filename = f"legacysurvey-{brick_name}-image-{band}.fits.fz"

    last_error = None

    for region in ["north", "south"]:
        url = (
            f"https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr8/"
            f"{region}/coadd/{bri}/{brick_name}/{filename}"
        )

        try:
            # print(f"Downloading/opening {url}")

            img = download_fits_cached(url)

            # print(
                # f"Loaded {band} image for {brick_name} "
                # f"with shape {img.shape}"
            # )

            return img

        except requests.HTTPError as e:
            last_error = e

            if e.response.status_code == 404:
                continue

            raise

    raise FileNotFoundError(
        f"Could not find {filename}"
    ) from last_error

def crop2d(img, x, y, size):
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    half = size // 2
    x = int(round(x))
    y = int(round(y))

    stamp = np.zeros((size, size), dtype=np.float32)

    y0 = max(0, y - half)
    y1 = min(img.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(img.shape[1], x + half)

    sy0 = y0 - (y - half)
    sx0 = x0 - (x - half)

    stamp[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)] = img[y0:y1, x0:x1]
    return stamp

def normalize_band(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)

    return x.astype(np.float32)

def read_image_fits(path):
    with fits.open(path, memmap=True) as hdul:
        hdul.info()

        for hdu in hdul:
            if hdu.data is None:
                continue

            data = np.asarray(hdu.data, dtype=np.float32)

            if data.ndim == 2:
                return data

            if data.ndim == 3:
                return data[0]

    raise ValueError(f"No 2D image found in {path}")


from concurrent.futures import ThreadPoolExecutor, as_completed

def process_brick(args):
    brick_name, group_df, bands, stamp_size = args

    band_images = {}
    for band in bands:
        try:
            band_images[band] = fetch_brick_band(brick_name, band)
        except FileNotFoundError:
            band_images[band] = None

    results = []

    for row in group_df.itertuples(index=False):
        x = getattr(row, "brick_x")
        y = getattr(row, "brick_y")
        object_id = getattr(row, "objectId")

        channels = []

        for band in bands:
            img = band_images[band]

            if img is None:
                stamp = np.zeros((stamp_size, stamp_size), dtype=np.float32)
            else:
                stamp = crop2d(img, x, y, stamp_size)
                stamp = normalize_band(stamp)

            channels.append(stamp)

        chw = np.stack(channels, axis=0)

        results.append({
            "object_id": int(object_id),
            "label": int(row.label),
            "ra": float(row.ra),
            "dec": float(row.dec),
            "data": chw,
        })

    return results


