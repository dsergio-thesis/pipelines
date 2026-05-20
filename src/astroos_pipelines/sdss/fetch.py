
import os
import io
import time
import json
import bz2
from urllib import response
import warnings
from collections import defaultdict
import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
from astropy.io.votable import parse_single_table
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.wcs.utils import pixel_to_skycoord

from abc import ABC, abstractmethod

import requests
from astropy.io import fits
from io import BytesIO

from scipy.ndimage import rotate

import sys
import importlib

# from astroos_pipelines.logger.logger import setup_logging
# importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
# import logging
# setup_logging()
# log = logging.getLogger(__name__)

class AstroosFetch(ABC):
    """
    Abstract base class for Astroosfetch clients.
    """

    def __init__(self, label_definitions):
        self.label_definitions = label_definitions

    @abstractmethod
    def fetch_images(self):
        pass


# ============================================================
# AstroosFetchSDSS
# ============================================================
class AstroosFetchSDSS(AstroosFetch):

    def __init__(self, label_definitions):
        super().__init__(label_definitions=label_definitions)
        self.label_definitions = label_definitions

    def fetch_images(self, positions, cache_dir, n=300, query_cache_file="sdss_query_cache.json"):

        os.makedirs(cache_dir, exist_ok=True)

        bands = ['u', 'g', 'r', 'i', 'z']
        rerun = 301
        grouped = defaultdict(list)
        coords = [SkyCoord(ra, dec, unit='deg') for ra, dec in positions]

        # Load or initialize query cache
        if os.path.exists(query_cache_file):
            with open(query_cache_file, "r") as f:
                query_cache = json.load(f)
        else:
            query_cache = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # print("Querying SDSS fields for each position...")
            for i, coord in enumerate(tqdm(coords, desc="Indexing fields")):
                key_str = f"{coord.ra.deg:.6f}_{coord.dec.deg:.6f}"

                if key_str in query_cache:
                    run, camcol, field = query_cache[key_str]
                else:
                    time.sleep(0.5)  # be nice to SDSS
                    try:
                        res = SDSS.query_region(coord, radius=5 * u.arcsec)
                        if res is None or len(res) == 0:
                            raise ValueError("No result")
                        run = int(res[0]['run'])
                        camcol = int(res[0]['camcol'])
                        field = int(res[0]['field'])
                        query_cache[key_str] = (run, camcol, field)
                    except Exception as e:
                        print(f"[ERROR] RA={coord.ra.deg}, Dec={coord.dec.deg}: {e}")
                        continue

                grouped[(run, camcol, field)].append((i, coord))

            # Save updated query cache
            with open(query_cache_file, "w") as f:
                json.dump(query_cache, f)

            output_images = [None] * len(positions)

            for key in tqdm(grouped, desc="Processing tiles"):
                run, camcol, field = key
                band_images = {}
                wcs = None

                for band in bands:
                    filename = f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
                    filepath = os.path.join(cache_dir, filename)

                    if not os.path.exists(filepath):
                        url = (
                            f"https://data.sdss.org/sas/dr17/eboss/photoObj/frames/"
                            f"{rerun}/{run}/{camcol}/{filename}"
                        )
                        try:
                            resp = requests.get(url)
                            resp.raise_for_status()
                            with open(filepath, "wb") as f:
                                f.write(resp.content)
                        except Exception as e:
                            print(f"[ERROR] Failed to download {url}: {e}")
                            continue

                    with open(filepath, "rb") as f:
                        raw = bz2.decompress(f.read())
                        hdul = fits.open(io.BytesIO(raw))
                        band_images[band] = hdul[0].data.astype(float)
                        if band == 'r':
                            wcs = WCS(hdul[0].header)
                        
                        # if band == 'r':
                        #     # display the image with a circle at the center
                        #     wcs = WCS(hdul[0].header)
                        #     x_center, y_center = wcs.world_to_pixel(grouped[(run, camcol, field)][0][1])
                        #     x_center = int(round(np.atleast_1d(x_center)[0]))
                        #     y_center = int(round(np.atleast_1d(y_center)[0]))
                        #     x0 = x_center - n // 2
                        #     y0 = y_center - n // 2
                        #     plt.figure(figsize=(10, 10))
                        #     plt.imshow(band_images[band], cmap='gist_ncar')
                        #     plt.scatter([x_center], [y_center], s=100, edgecolor='red', facecolor='none')
                        #     plt.title(f"Band {band}")
                        #     plt.show()
                        #     print(f"Cutout top-left corner - x0: {x0}, y0: {y0}")


                if wcs is None:
                    # print(f"[WARNING] Missing r-band WCS for {key}")
                    continue

                for idx, coord in grouped[key]:
                    x_center, y_center = wcs.world_to_pixel(coord)
                    x_center = int(round(np.atleast_1d(x_center)[0]))
                    y_center = int(round(np.atleast_1d(y_center)[0]))
                    x0 = x_center - n // 2
                    y0 = y_center - n // 2

                    cutout = np.zeros((5, n, n), dtype=float)
                    for i, band in enumerate(bands):
                        arr = band_images.get(band)
                        if arr is None:
                            continue
                        h, w_img = arr.shape
                        x1 = max(x0, 0)
                        x2 = min(x0 + n, w_img)
                        y1 = max(y0, 0)
                        y2 = min(y0 + n, h)
                        dx1 = x1 - x0
                        dx2 = dx1 + (x2 - x1)
                        dy1 = y1 - y0
                        dy2 = dy1 + (y2 - y1)
                        cutout[i, dy1:dy2, dx1:dx2] = arr[y1:y2, x1:x2]

                    output_images[idx] = cutout

        return output_images



# ============================================================
# AstroosFetchSDSSManualCutout_ImageOnly
# ============================================================
class AstroosFetchSDSSManualCutout_ImageOnly(AstroosFetch):

    def __init__(self, df, dir, image_url_format_string, n=300):
        super().__init__(label_definitions=None)
        self.df = df
        self.image_url_format_string = image_url_format_string
        self.n = n
        self.dir = dir
    def fetch_images(self):

        df = self.df
        image_url_format_string = self.image_url_format_string
        n = self.n
        n = 100

        output_images = []
        df_i = 0
        bands = ['u', 'g', 'r', 'i', 'z']

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            output_images = np.zeros((len(df), 5, n, n), dtype=float)
            output_band_bounds = np.zeros((len(df), 5, 4), dtype=float)

            for row in tqdm(df.itertuples(), total=len(df), desc="Downloading images"):
                ra = row.ra
                dec = row.dec
                coord = SkyCoord(ra, dec, unit='deg')

                label = row.label
                band_images = {}
                wcs = None

                band_index = 0
                for band in ['u', 'g', 'r', 'i', 'z']:
                    
                    rerun = getattr(row, f"{band}_rerun")
                    run = getattr(row, f"{band}_run")
                    run06d = getattr(row, f"{band}_run06d")
                    camcol = getattr(row, f"{band}_camcol")
                    field = getattr(row, f"{band}_field")
                    field04d = getattr(row, f"{band}_field04d")

                    image_url = image_url_format_string.format(
                        rerun=rerun,
                        run=run,
                        camcol=camcol,
                        band=band,
                        run06d=run06d,
                        field04d=field04d
                    )
                    log.debug(f"Fetching image for RA: {ra}, Dec: {dec}, Label: {label} - Band: {band} from URL: {image_url}")

                    filepath = f"{self.dir}/{ra}_{dec}_{band}.bz2"
                    try:
                        time.sleep(1) # be nice to SDSS
                        resp = requests.get(image_url)
                        resp.raise_for_status()

                        # save to bytes instead of file
                        image_data = io.BytesIO(resp.content)
                        # with open(filepath, "wb") as f:
                        #     f.write(resp.content)
                    except Exception as e:
                        print(f"Failed to download {image_url}: {e}")
                        continue

                    # with open(filepath, "rb") as f:
                    with image_data as f:
                        raw = bz2.decompress(f.read())
                        hdul = fits.open(io.BytesIO(raw))
                        band_images[band_index] = hdul[0].data.astype(float)


                        # if band == 'r':
                        #     # display the image with a circle at the center
                        #     wcs = WCS(hdul[0].header)
                        #     x_center, y_center = wcs.world_to_pixel(coord)
                        #     x_center = int(round(np.atleast_1d(x_center)[0]))
                        #     y_center = int(round(np.atleast_1d(y_center)[0]))
                        #     x0 = x_center - n // 2
                        #     y0 = y_center - n // 2
                        #     plt.figure(figsize=(10, 10))
                        #     plt.imshow(band_images[band_index], cmap='gist_ncar')
                        #     plt.scatter([x_center], [y_center], s=100, edgecolor='red', facecolor='none')
                        #     plt.title(f"Band {band}")
                        #     plt.show()
                        #     print(f"Center pixel coordinates for RA: {ra}, Dec: {dec} - x: {x_center}, y: {y_center}")
                        #     print(f"Cutout top-left corner - x0: {x0}, y0: {y0}")

                        # print(f"Loaded band {band} image with shape {band_images[band_index].shape}")
                        band_index += 1
                        if band == 'r':
                            wcs = WCS(hdul[0].header)
                    
                x_center, y_center = wcs.world_to_pixel(coord)
                x_center = int(round(np.atleast_1d(x_center)[0]))
                y_center = int(round(np.atleast_1d(y_center)[0]))
                x0 = x_center - n // 2
                y0 = y_center - n // 2

                band_bounds = np.zeros((5, 4), dtype=float)

                cutout = np.zeros((5, n, n), dtype=float)
                for i, band in enumerate(bands):
                    arr = band_images.get(i)
                    if arr is None:
                        continue
                    h, w_img = arr.shape
                    x1 = max(x0, 0)
                    x2 = min(x0 + n, w_img)
                    y1 = max(y0, 0)
                    y2 = min(y0 + n, h)
                    dx1 = x1 - x0
                    dx2 = dx1 + (x2 - x1)
                    dy1 = y1 - y0
                    dy2 = dy1 + (y2 - y1)

                    each_band_cutout = arr[y1:y2, x1:x2]

                    # Compute rotation from CD/PC matrix
                    cd = wcs.wcs.cd
                    if cd is None:
                        cd = np.dot(np.diag(wcs.wcs.cdelt), wcs.wcs.pc)
                    
                    # rotation angle in degrees (negative to rotate sky North up)
                    theta = np.arctan2(cd[0,1], cd[0,0])
                    theta_deg = -np.degrees(theta)

                    # Rotate the image
                    image_rot = rotate(each_band_cutout, theta_deg, reshape=False, order=3)

                    # Check RA axis: ensure East is left
                    # ra_min, dec_min = wcs.wcs_pix2world(0, 0, 0)
                    # ra_max, _ = wcs.wcs_pix2world(each_band_cutout.shape[1]-1, 0, 0)
                    ra_min, dec_min = wcs.wcs_pix2world(0, 0, 0)  # top-left
                    ra_max, dec_top = wcs.wcs_pix2world(each_band_cutout.shape[1]-1, each_band_cutout.shape[0]-1, 0)

                    if ra_max < ra_min:
                        image_rot = np.fliplr(image_rot)  # flip horizontally
                        ra_min, ra_max = ra_max, ra_min  # swap
                    
                    # Check Dec axis: ensure North is up
                    # dec_min, ra_min = wcs.wcs_pix2world(0, 0, 0)
                    # dec_max, _ = wcs.wcs_pix2world(0, each_band_cutout.shape[0]-1, 0)
                    _, dec_bottom = wcs.wcs_pix2world(0, each_band_cutout.shape[0]-1, 0)

                    if dec_bottom > dec_top:
                        image_rot = np.flipud(image_rot)  # flip vertically
                        dec_bottom, dec_top = dec_top, dec_bottom  # swap

                    # for some reason this is needed again
                    image_rot = np.flipud(image_rot)
                    # image_rot = np.fliplr(image_rot)

                    cutout[i, dy1:dy2, dx1:dx2] = image_rot

                    log.debug(f"Band {band} bounds - RA: [{ra_min}, {ra_max}], Dec: [{dec_bottom}, {dec_top}]")
                    band_bounds[i] = np.array([ra_min, ra_max, dec_bottom, dec_top])

                # print(f"df_i: {df_i}, ra: {ra}, dec: {dec}, label: {label}")
                output_images[df_i] = cutout
                output_band_bounds[df_i] = band_bounds


                df_i += 1
    
        # output_images = np.array(output_images)
        print(f"Output images shape: {output_images.shape}")

        return output_images, df['label'].values, output_band_bounds



# ============================================================
# AstroosFetchManualFitsCutout
# ============================================================
class AstroosFetchManualFitsCutout(AstroosFetch):

    def __init__(self, df, dir, image_url_format_string, dataset, n=100):
        super().__init__(label_definitions=None)
        self.df = df
        self.image_url_format_string = image_url_format_string
        self.n = n
        self.dir = dir
        self.dataset = dataset

    def fetch_images(self):
        df = self.df
        image_url_format_string = self.image_url_format_string
        n = self.n
        dataset = self.dataset
        df_i = 0
        output_band_bounds = np.zeros((len(df), 5, 4), dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            image_cutouts = np.zeros((len(df), 5, n, n), dtype=float)

            progress_bar = tqdm(df.itertuples(), total=len(df), desc="Constructing Dataset...")

            for row in progress_bar:

                ra = row.ra
                dec = row.dec
                coord = SkyCoord(ra, dec, unit='deg')

                if dataset._contains(row.main_id):
                    log.debug(f"({ra}, {dec}) Label: {row.label}. Skipping ")
                    df_i += 1
                    continue
                else:
                    print(f"Downloading ({ra}, {dec}) Label: {row.label}")
                label = row.label
                band_hdu_cutouts = []
                wcs = None
                bands = ['u', 'g', 'r', 'i', 'z']

                for band in bands:
                    
                    rerun = getattr(row, f"{band}_rerun")
                    run = getattr(row, f"{band}_run")
                    run06d = getattr(row, f"{band}_run06d")
                    camcol = getattr(row, f"{band}_camcol")
                    field = getattr(row, f"{band}_field")
                    field04d = getattr(row, f"{band}_field04d")

                    image_url = image_url_format_string.format(
                        rerun=rerun,
                        run=run,
                        camcol=camcol,
                        band=band,
                        run06d=run06d,
                        field04d=field04d
                    )
                    log.debug(f"Fetching image for RA: {ra}, Dec: {dec}, Label: {label} - Band: {band} from URL: {image_url}")

                    try:
                        time.sleep(1) # be nice to SDSS
                        response = requests.get(image_url)
                        response.raise_for_status()

                        image_data = io.BytesIO(response.content)

                    except Exception as e:
                        print(f"Failed to download {image_url}: {e}")
                        continue

                    with image_data as f:
                        raw = bz2.decompress(f.read())
                        hdul = fits.open(io.BytesIO(raw))

                        cutout = Cutout2D(
                            hdul[0].data,
                            position=coord,
                            size=(n, n),
                            mode='partial',
                            fill_value=0.0,
                            wcs=WCS(hdul[0].header)
                        )
                        # print(cutout.data)
                        wcs = cutout.wcs

                        band_hdu_cutouts.append(cutout)

                        image_cutout = cutout.data.astype(float)
                        image_cutouts[df_i, bands.index(band), :, :] = image_cutout

                bounds = AstroosFetchManualFitsCutout.get_cutout_bounds(band_hdu_cutouts[0])

                hdul = fits.HDUList([fits.PrimaryHDU()])
                hdu = fits.ImageHDU(data=image_cutouts[df_i], header=wcs.to_header(), name="CUTOUTS")

                # 'main_id', 'rvz_redshift', 'galdim_majaxis', 'galdim_minaxis', 'galdim_angle
                hdu.header['label'] = label
                hdu.header['ra'] = ra
                hdu.header['dec'] = dec
                hdu.header['main_id'] = row.main_id
                try:
                    hdu.header['rvz_redshift'] = row.rvz_redshift
                except ValueError:
                    hdu.header['rvz_redshift'] = '-999'
                # hdu.header['majaxis'] = row.galdim_majaxis
                # hdu.header['minaxis'] = row.galdim_minaxis
                # hdu.header['angle'] = row.galdim_angle
                hdu.header['min_ra'] = bounds[0]
                hdu.header['max_ra'] = bounds[1]
                hdu.header['min_dec'] = bounds[2]
                hdu.header['max_dec'] = bounds[3]

                output_band_bounds[df_i] = bounds

                hdul.append(hdu)
                dataset.append(hdul)

                df_i += 1

        return image_cutouts, df['label'].values, output_band_bounds


    @staticmethod
    def get_cutout_bounds(cutout):
        """
        Returns (ra_min, ra_max, dec_min, dec_max) for a Cutout2D object.
        """
        h, w = cutout.data.shape

        # corners in cutout pixel coordinates
        corners = [
            (0, 0),           # bottom-left
            (w-1, 0),         # bottom-right
            (0, h-1),         # top-left
            (w-1, h-1)        # top-right
        ]

        sky = pixel_to_skycoord(
            [x for x, y in corners],
            [y for x, y in corners],
            cutout.wcs
        )

        ras  = sky.ra.deg
        decs = sky.dec.deg

        return ras.min(), ras.max(), decs.min(), decs.max()
