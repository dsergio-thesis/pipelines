
import sys
import os
import pandas as pd

import numpy as np
import torch

from tqdm import tqdm
from astropy.coordinates import SkyCoord
from io import BytesIO
from collections import defaultdict

from astropy.table import Table, vstack
from astropy import units as u
import time

import requests
from PIL import Image
from io import BytesIO
import warnings
import importlib

from utils.formatting_utils import ascii_kv_table
importlib.reload(sys.modules['utils.formatting_utils'])

from astroos_pipelines.pipelines import DataPipelineStage
from astroos_pipelines.sdss.query import AstroosQuerySDSS
from astroos_pipelines.sdss.fetch import AstroosFetchManualFitsCutout

importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.sdss.query'])
importlib.reload(sys.modules['astroos_pipelines.sdss.fetch'])

import bz2, io
from astropy.io import fits
import requests
from astropy.wcs import WCS
from astropy.io import fits

from astroquery.simbad import Simbad
from astroquery.sdss import SDSS

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class StageCatalogSDSS(DataPipelineStage):
    """
    Data pipeline stage for cataloging SDSS data.
    """

    def __init__(self):
        super().__init__(stage_name="catalog", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return True

    def run(self):

        query_coords = self.pipeline.metadata.get('query_coords')
        query_radius = self.pipeline.metadata.get('query_radius')

        sdss_client = AstroosQuerySDSS(root_dir=self.stage_dir)

        dec_min = max(query_coords.dec.deg - query_radius.to(u.deg).value, -90)
        dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

        delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
        ra_min = (query_coords.ra.deg - delta_ra) % 360
        ra_max = (query_coords.ra.deg + delta_ra) % 360

        sdss_client.scan_TAP(
            ra_min=ra_min, 
            ra_max=ra_max, 
            dec_min=dec_min, 
            dec_max=dec_max,
            ra_offset= (ra_max - ra_min) / 10, 
            limit=self.pipeline.max_records)


class StageCatalogSDSS_V2(DataPipelineStage):
    """
    Data pipeline stage for cataloging SDSS data.
    """

    def __init__(self):
        super().__init__(stage_name="catalog", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return True

    def run(self):

        query_coords = self.pipeline.metadata.get('query_coords')
        query_radius = self.pipeline.metadata.get('query_radius')

        sdss_client = AstroosQuerySDSS(root_dir=self.stage_dir)

        dec_min = max(query_coords.dec.deg - query_radius.to(u.deg).value, -90)
        dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

        delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
        ra_min = (query_coords.ra.deg - delta_ra) % 360
        ra_max = (query_coords.ra.deg + delta_ra) % 360

        self.output = sdss_client.scan(
            self.pipeline, 
            ra_min=180, ra_max=184, dec_min=dec_min, dec_max=dec_max,
            limit=self.pipeline.max_records)
        # first 5 rows
        # self.output = self.output[:5]


class StageFetchSDSS_V2_ManualCutout(DataPipelineStage):
    """Data pipeline stage for fetching SDSS data."""
    def __init__(self):
        super().__init__(stage_name="fetch", requires_stage_dir=True)

    def _validate_prev_stage(self):
        if not self.prev_stage or not self.prev_stage.output:
            raise ValueError("Previous stage output is missing.")
        df, image_url_format_string = self.prev_stage.output
        if df is None or image_url_format_string is None:
            raise ValueError("Previous stage output is invalid.")
        
        required_columns = ['ra', 'dec', 'label']
        bands = ['u', 'g', 'r', 'i', 'z']
        for band in bands:
            required_columns.append(f"{band}_rerun")
            required_columns.append(f"{band}_run")
            required_columns.append(f"{band}_run06d")
            required_columns.append(f"{band}_camcol")
            required_columns.append(f"{band}_field")
            required_columns.append(f"{band}_field04d")

        if not all(col in df.columns for col in required_columns):
            print(f"Missing columns in DataFrame: {[col for col in required_columns if col not in df.columns]}")
            raise ValueError(f"DataFrame from previous stage must contain columns: {required_columns}. Actual columns: {df.columns.tolist()}")

        return True

    def run(self):
        # read the positions from the previous stage

        df, image_url_format_string = self.prev_stage.output

        print(image_url_format_string)

        astroosFetch = AstroosFetchSDSSManualCutout(
            df=df,
            dir=self.stage_dir,
            image_url_format_string=image_url_format_string
        )

        output, labels, output_band_bounds = astroosFetch.fetch_images()

        self.output = output, labels, output_band_bounds

        images_tensor = torch.tensor(output, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        torch.save(images_tensor, f"{self.pipeline.pipeline_dir}/X_train.pt")
        torch.save(labels_tensor, f"{self.pipeline.pipeline_dir}/y_train.pt")
        print(f"Saved file: {self.pipeline.pipeline_dir}/X_train.pt and {self.pipeline.pipeline_dir}/y_train.pt")

        self.pipeline.X_train_filename = f"{self.pipeline.pipeline_dir}/X_train.pt"
        self.pipeline.y_train_filename = f"{self.pipeline.pipeline_dir}/y_train.pt"


class StageFetchSDSS_V3_ManualCutout(DataPipelineStage):
    """
    Data pipeline stage for fetching SDSS data.
    """

    def __init__(self, dataset):
        super().__init__(stage_name="fetch", requires_stage_dir=True)
        self.dataset = dataset

    def _validate_prev_stage(self):
        if not self.prev_stage or not self.prev_stage.output:
            raise ValueError("Previous stage output is missing.")
        df, image_url_format_string = self.prev_stage.output
        if df is None or image_url_format_string is None:
            raise ValueError("Previous stage output is invalid.")
        
        required_columns = ['ra', 'dec', 'label']
        bands = ['u', 'g', 'r', 'i', 'z']
        for band in bands:
            required_columns.append(f"{band}_rerun")
            required_columns.append(f"{band}_run")
            required_columns.append(f"{band}_run06d")
            required_columns.append(f"{band}_camcol")
            required_columns.append(f"{band}_field")
            required_columns.append(f"{band}_field04d")

        if not all(col in df.columns for col in required_columns):
            log.error(f"Missing columns in DataFrame: {[col for col in required_columns if col not in df.columns]}")
            raise ValueError(f"DataFrame from previous stage must contain columns: {required_columns}. Actual columns: {df.columns.tolist()}")

        return True

    def run(self):
        # read the positions from the previous stage

        df, image_url_format_string = self.prev_stage.output

        log.info(f"Image URL format string: {image_url_format_string}")

        astroosFetch = AstroosFetchManualFitsCutout(
            df=df,
            dir=self.stage_dir,
            image_url_format_string=image_url_format_string,
            dataset=self.dataset
        )

        output, labels, output_band_bounds = astroosFetch.fetch_images()

        self.output = output, labels, output_band_bounds

        # images_tensor = torch.tensor(output, dtype=torch.float32)
        # labels_tensor = torch.tensor(labels, dtype=torch.int64)
        # torch.save(images_tensor, f"{self.pipeline.dataset.dir}/X_train.pt")
        # torch.save(labels_tensor, f"{self.pipeline.dataset.dir}/y_train.pt")
        # print(f"Saved file: {self.pipeline.dataset.dir}/X_train.pt and {self.pipeline.dataset.dir}/y_train.pt")

        # self.pipeline.X_train_filename = f"{self.pipeline.dataset.dir}/X_train.pt"
        # self.pipeline.y_train_filename = f"{self.pipeline.dataset.dir}/y_train.pt"


class StageFetchSDSS_V2_AutoCutout(DataPipelineStage):
    """Data pipeline stage for fetching SDSS data."""
    def __init__(self, dataset):
        super().__init__(stage_name="fetch", requires_stage_dir=True)
        self.dataset = dataset

    def _validate_prev_stage(self):
        if not self.prev_stage or not self.prev_stage.output:
            raise ValueError("Previous stage output is missing.")
        df, image_url_format_string = self.prev_stage.output
        if df is None or image_url_format_string is None:
            raise ValueError("Previous stage output is invalid.")
        
        required_columns = ['ra', 'dec', 'label']

        if not all(col in df.columns for col in required_columns):
            print(f"Missing columns in DataFrame: {[col for col in required_columns if col not in df.columns]}")
            raise ValueError(f"DataFrame from previous stage must contain columns: {required_columns}")

        return True

    def run(self):
        # read the positions from the previous stage

        df, _ = self.prev_stage.output

        bands = ['u', 'g', 'r', 'i', 'z']
        n = 100
        labels = df['label'].tolist()

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            output_images = np.zeros((len(df), 5, n, n), dtype=float)
            output_image_bounds = np.zeros((len(df), 5, 4), dtype=float)  # min_ra, max_ra, min_dec, max_dec

            for row in tqdm(df.itertuples(), total=len(df), desc="Downloading SDSS Cutout Images"):
                ra = row.ra
                dec = row.dec
                scale = 0.396  # arcsec/pixel
                width = height = int(n)
                band_images = np.zeros((5, n, n), dtype=float)
                band_bounds = np.zeros((5, 4), dtype=float)  # min_ra, max_ra, min_dec, max_dec

                for i, band in enumerate(['u', 'g', 'r', 'i', 'z']):

                    opt = band.upper()
                    url = (
                        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
                        f"?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}&opt={opt}"
                    )

                    min_ra = ra - (scale * width) / 3600 / 2
                    max_ra = ra + (scale * width) / 3600 / 2
                    min_dec = dec - (scale * height) / 3600 / 2
                    max_dec = dec + (scale * height) / 3600 / 2

                    band_bounds[i] = [
                        min_ra,
                        max_ra,
                        min_dec,
                        max_dec
                    ]

                    # print("Fetching URL:", url)
                    response = requests.get(url)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content)).convert('L')
                        band_images[i] = np.array(img)
                    else:
                        print(f"[ERROR] Failed to download {url}: {response.status_code}")

                output_images[row.Index] = band_images
                output_image_bounds[row.Index] = band_bounds

        print(f"Downloaded {len(output_images)} SDSS cutout images. shape: {output_images.shape}")
        # print(f"Image dimensions shape: {output_image_bounds}")

        self.output = output_images, labels, output_image_bounds

        images_tensor = torch.tensor(output_images, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        self.pipeline.X_train_filename = f"{self.pipeline.pipeline_dir}/X_train.pt"
        self.pipeline.y_train_filename = f"{self.pipeline.pipeline_dir}/y_train.pt"

        torch.save(images_tensor, self.pipeline.X_train_filename)
        torch.save(labels_tensor, self.pipeline.y_train_filename)
        print(f"Saved file: {self.pipeline.X_train_filename} and {self.pipeline.y_train_filename}")


class StageFilterCatalogSDSS(DataPipelineStage):
    """
    Data pipeline stage for filtering SDSS catalog results.
    """

    def __init__(self, pipeline_dir, stage_dir, pipeline):
        super().__init__(stage_name=stage_dir, pipeline_dir=pipeline_dir, stage_dir=stage_dir, pipeline=pipeline)

    def run(self):

        print("Filtering SDSS catalog results...")

        

        sdss_catalog = AstroosCatalogSDSS(dir=self.pipeline_dir, 
                                          catalog_dir=self.prev_stage_dir, 
                                          pipeline=self.pipeline, 
                                          max_records=self.pipeline.max_records)

        sdss_catalog.load_catalog()
        pos, labels, rows = sdss_catalog.filter_catalog()
        print(f"Filtered positions: {pos}")
        print(f"Filtered labels: {labels}")
        rows_df = pd.DataFrame(rows)
        df = pd.DataFrame({
            "position": pos, 
            "label": labels, 
            "main_id": rows_df['main_id'] if 'main_id' in rows_df else [None]*len(pos),
            "rvz_redshift": rows_df['rvz_redshift'] if 'rvz_redshift' in rows_df else [None]*len(pos),
            "galdim_majaxis": rows_df['galdim_majaxis'] if 'galdim_majaxis' in rows_df else [None]*len(pos),
            "galdim_minaxis": rows_df['galdim_minaxis'] if 'galdim_minaxis' in rows_df else [None]*len(pos),
            "galdim_angle": rows_df['galdim_angle'] if 'galdim_angle' in rows_df else [None]*len(pos),
        })
        df.to_csv(f"{self.stage_dir}/positions_labels.csv", index=False)

        self.pipeline.filtered_positions = pos
        self.pipeline.filtered_labels = labels
        self.pipeline.filtered_rows = rows


class StageFetchSDSS(DataPipelineStage):
    """
    Data pipeline stage for fetching SDSS images.
    """

    def __init__(self, chunk_size=100):
        super().__init__(stage_name="fetch", requires_stage_dir=True)
        self.chunk_size = chunk_size

    def _validate_prev_stage(self):
        return True

    def run(self):

        print("Fetching SDSS images...")

        df, image_url_format_string = self.prev_stage.output
        ra = df['ra'].to_list()
        dec = df['dec'].to_list()
        labels_list = df['label'].to_list()

        positions = list(zip(ra, dec))

        print(f"Total positions to fetch: {len(positions)} total labels: {len(labels_list)}")
        print(positions)
        print(labels_list)

        astroos_fetch = AstroosFetchSDSS(label_definitions=self.pipeline.label_definitions)


        # break positions into chunks
        chunk_size = self.chunk_size
        positions_chunked = [positions[i:i + chunk_size] for i in range(0, len(positions), chunk_size)]
        starting_indices = [i for i in range(0, len(positions), chunk_size)]
        print(f"Total chunks: {len(positions_chunked)}")

        for i, chunk in enumerate(positions_chunked):
            print(f"Chunk {i}: {len(chunk)} positions, starting position: {starting_indices[i]}")

        labels_list_chunked = [labels_list[i:i + chunk_size] for i in range(0, len(labels_list), chunk_size)]
        print(f"Total labels chunks: {len(labels_list_chunked)}")
        for i, chunk in enumerate(labels_list_chunked):
            print(f"Labels Chunk {i}: {len(chunk)} labels")


        suffix = self.pipeline.pipeline_name

        for i, chunk in enumerate(positions_chunked):

            print(f"Chunk {i}: {len(chunk)} positions, starting position")
            N_chunk = len(chunk)

            images_chunk = astroos_fetch.fetch_images(positions=chunk, cache_dir=self.stage_dir, n=300)

            to_delete = []
            for j in range(len(images_chunk)):
                if images_chunk[j] is None:
                    to_delete.append(j)

            for j in reversed(to_delete):
                del images_chunk[j]
            print(f"Total images_chunk: {len(images_chunk)}")

            for j in reversed(to_delete):
                del labels_list_chunked[i][j]

            images_array_chunk = np.array(images_chunk)
            print(f"Total images: {len(images_array_chunk)}")
            print(f"labels_list_chunked[i]: {len(labels_list_chunked[i])}")

            galaxies_tensor = torch.tensor(images_array_chunk, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list_chunked[i], dtype=torch.int64)

            if i == 0:
                torch.save(galaxies_tensor, f"{self.stage_dir}/X_train_{suffix}.pt")
                torch.save(labels_tensor, f"{self.stage_dir}/y_train_{suffix}.pt")
                print(f"Saved chunk {i} to file: X_train_{suffix}.pt and y_train_{suffix}.pt")
            else:
                galaxies_tensor_existing = torch.load(f"{self.stage_dir}/X_train_{suffix}.pt", map_location=torch.device('cpu'))
                labels_tensor_existing = torch.load(f"{self.stage_dir}/y_train_{suffix}.pt", map_location=torch.device('cpu'))

                appended_tensor = torch.cat((galaxies_tensor_existing, galaxies_tensor), dim=0)
                torch.save(appended_tensor, f"{self.stage_dir}/X_train_{suffix}.pt")

                appended_labels = torch.cat((labels_tensor_existing, labels_tensor), dim=0)
                torch.save(appended_labels, f"{self.stage_dir}/y_train_{suffix}.pt")
            print(f"Saved chunk {i} to file: X_train_{suffix}.pt and y_train_{suffix}.pt")

        # Load the saved tensors
        if (os.path.exists(f"{self.stage_dir}/X_train_{suffix}.pt") and
            os.path.exists(f"{self.stage_dir}/y_train_{suffix}.pt")):
            X_train = torch.load(f"{self.stage_dir}/X_train_{suffix}.pt", map_location=torch.device('cpu'))
            y_train = torch.load(f"{self.stage_dir}/y_train_{suffix}.pt", map_location=torch.device('cpu'))
            print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")

            self.pipeline.X_train_filename = f"{self.stage_dir}/X_train_{suffix}.pt"
            self.pipeline.y_train_filename = f"{self.stage_dir}/y_train_{suffix}.pt"
        else:
            print("No images were fetched.")
            self.pipeline.X_train_filename = None
            self.pipeline.y_train_filename = None

