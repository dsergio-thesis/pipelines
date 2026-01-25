
from abc import ABC, abstractmethod
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

if (sys.modules.get('src.astroos.query') is not None):
    del sys.modules['src.astroos.query']
if (sys.modules.get('src.astroos.catalogs') is not None):
    del sys.modules['src.astroos.catalogs']
if (sys.modules.get('src.astroos.fetch') is not None):
    del sys.modules['src.astroos.fetch']

from src.astroos.catalogs import AstroosCatalogSDSS, \
    AstroosCatalogLSST
from src.astroos.query import AstroosQueryNED, \
    AstroosQuerySimbad, AstroosQuerySDSS, \
    AstroQueryUtils as aq_utils, AstroosQueryLSST
from src.astroos.fetch import AstroosFetchSDSS, \
    AstroosFetchLSST, AstroosFetchSDSSManualCutout, \
    AstroosFetchManualFitsCutout

import bz2, io
from astropy.io import fits
import requests
from astropy.wcs import WCS

if (sys.modules.get('src.astroos.utils.rsp_utils') is not None): 
    del sys.modules['src.astroos.utils.rsp_utils']

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    from lsst.rsp.utils import get_pyvo_auth
    import lsst.geom as geom


    # other LSST dependencies
    from pyvo.dal.adhoc import DatalinkResults
    from astropy.time import Time
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery


    rsp_mode = True
    print("LSST RSP mode enabled.")
except ImportError:
    print("LSST RSP mode disabled.")


# ============================================================
# Pipeline
# ============================================================
class Pipeline(ABC):
    """
    Abstract base class for data pipelines.
    """

    def __init__(self, name, metadata={}, max_records=None, minor_version=None):
        self.stages = [StageInfo()]
        self.stages[0].stage_name = "Initialization"
        self.stages[0].stage_index = 0

        self._major_version = 2
        self.pipeline_name, self.pipeline_dir = self._construct_name(name, attr={}, minor_version=minor_version)

        self.max_records = max_records
        self.credentials_file = None
    
        os.makedirs(self.pipeline_dir, exist_ok=True)
        self.stages_added = False
        self.metadata = metadata
        self.output = None
        print(f"Data pipeline created: {self.pipeline_name}")

    def _construct_name(self, 
                        name: str, 
                        attr: dict[str, str] = None,
                        minor_version: int = None
                        ) -> str:
        """ Construct a unique name for the pipeline based on its attributes. """

        if minor_version is not None:
            pipeline_name = f"{name}_v{self._major_version}.{minor_version}"
        else:
            pipeline_name = f"{name}_v{self._major_version}.0"

        for key, value in attr.items():
            pipeline_name += f"__{key}_{value}"

        pipeline_dir = f"./_pipelines/{pipeline_name}"
        os.makedirs(pipeline_dir, exist_ok=True)
        # find highest numbered subdir v2.x
        existing_dirs = [d for d in os.listdir(f"./_pipelines") if d.startswith(name)]
        if existing_dirs and minor_version is None:
            print([d.split("_v2.") for d in existing_dirs])
            highest_version = max([int(d.split("_v2.")[-1]) for d in existing_dirs])
            pipeline_name = f"{name}_v2.{highest_version + 1}"
            pipeline_dir = f"./_pipelines/{name}_v2.{highest_version + 1}"

        return pipeline_name, pipeline_dir

    @abstractmethod
    def prepare_pipeline(self):
        pass

    def add_stages(self, stages):

        if self.stages_added:
            raise RuntimeError("Stages have already been added to the pipeline.")

        stage_index = self.stages[-1].stage_index + 1 if self.stages else 0
        for stage in stages:

            stage.stage_index = stage_index
            stage_index += 1
            stage.pipeline = self
            stage.pipeline_dir = self.pipeline_dir
            stage.prev_stage = self.stages[-1]
            stage.prev_stage_dir = self.stages[-1].stage_dir

            if stage.requires_stage_dir is True:
                stage.stage_dir = os.path.join(stage.pipeline.pipeline_dir, stage.stage_name)
                os.makedirs(stage.stage_dir, exist_ok=True)
            
            print(f"Adding stage:\n{stage}")
            self.stages.append(stage)

            # stage = StageInfo()
            # stage.stage_index = stage_index
            # stage.prev_stage = self.stages[-1]
            # stage.prev_stage_dir = self.stages[-1].stage_dir
            # stage_index += 1
            # self.stages.append(stage)
        
        self.stages_added = True

    def run_pipeline(self):
        """
        Run prepare_pipeline(), then run() each stage of the pipeline in sequence.
        """
        
        self.prepare_pipeline()

        print(f"---------------- Running data pipeline ---------------- \n"
              f"directory: {self.pipeline_dir}...")
        for stage in self.stages:
            print(f"{stage.stage_index} Running stage: {stage.stage_name}")
            if not stage._validate_prev_stage():
                raise RuntimeError(f"Invalid inputs for stage {stage.stage_name}.")
            stage.run()
            print()
        print("Data pipeline completed.")

        self.output = self.stages[-1].output
    
    def clear_pipeline(self):
        """ Clear the pipeline directory if it exists. """
        if os.path.exists(self.pipeline_dir):
            import shutil
            # shutil.rmtree(self.pipeline_dir)
            print(f"Cleared pipeline directory: {self.pipeline_dir}")

    def __repr__(self):
        s = f"Pipeline(pipeline_name={self.pipeline_name}, max_records={self.max_records})"
        return s


# ============================================================
# PipelineClassification
# ============================================================
class PipelineClassification(Pipeline):
    """
    Data pipeline for classification data. 

    Attributes:
        
    """

    def __init__(self,
                 name,
                 max_records,
                 dataset,
                 metadata={},
                 minor_version=None
                 ):
        """ 
        Initialize the data pipeline. 
        
        Parameters:
        """
        super().__init__(name=name, metadata=metadata, max_records=max_records, minor_version=minor_version)
        self.dataset = dataset
        self.X_train_filename = None
        self.y_train_filename = None

    def prepare_pipeline(self):
        pass



# ============================================================
# PipelineDummy
# ============================================================
class PipelineDummy(Pipeline):
    """
    Dummy data pipeline for testing.
    """

    def __init__(self, name, max_records=1, metadata={}, minor_version=None):
        super().__init__(name=name, metadata=metadata, max_records=max_records, minor_version=minor_version)
        print("Initialized Dummy Pipeline.")

    def _validate_prev_stage(self):
        return True
    
    def run(self):
        print("Running Dummy Pipeline...")

    def prepare_pipeline(self):
        pass


# ============================================================
# DataPipelineStage
# ============================================================
class DataPipelineStage(ABC):
    """
    Abstract base class for data pipeline stages.
    """

    def __init__(self, stage_name, requires_stage_dir=False):
        self.stage_name = stage_name
        self.requires_stage_dir = requires_stage_dir
        self.stage_dir = None
        self.pipeline = None
        self.prev_stage = None
        self.prev_stage_dir = None
        self.output = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _validate_prev_stage(self):
        pass

    def __repr__(self):
        s = f"{self.stage_index} DataPipelineStage(stage_name={self.stage_name}, requires_stage_dir={self.requires_stage_dir})\n" \
            f" - requires_stage_dir={self.requires_stage_dir})\n" \
            f" - stage_dir={self.stage_dir}\n" \
            f" - pipeline={self.pipeline}\n" \
            f" - prev_stage={self.prev_stage.stage_name if self.prev_stage else None}\n" \
            f" - prev_stage_dir={self.prev_stage_dir}\n" \
            f" - output={self.output}\n"
        return s


# ============================================================
# StageInfo
# ============================================================
class StageInfo(DataPipelineStage):
    """
    Data pipeline stage for reporting info.
    """

    def __init__(self):
        super().__init__(stage_name="info")

    def _validate_prev_stage(self):
        return True

    def run(self):
        s = f"{self.stage_index} Stage {self.stage_name}"
        if self.prev_stage is not None:
            s += f" reporting on {self.prev_stage.stage_name}: output: {self.prev_stage.output}\n"
        print(s)


# ============================================================
# StageCatalogSDSS
# ============================================================
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


# ============================================================
# StageCatalogSDSS_V2
# ============================================================
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


# ============================================================
# StageCatalogLSST
# ============================================================
class StageCatalogLSST(DataPipelineStage):
    """
    Data pipeline stage for cataloging data.
    """

    def __init__(self):
        super().__init__(stage_name="catalog", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return True
    
    def run(self):

        query_coords = self.pipeline.metadata.get('query_coords')
        query_radius = self.pipeline.metadata.get('query_radius')

        client = AstroosQueryLSST(root_dir=self.stage_dir, 
                                  credentials_file=self.pipeline.credentials_file,
                                  max_records=self.pipeline.max_records)

        dec_min = max(self.pipeline.metadata.get('query_coords').dec.deg - self.pipeline.metadata.get('query_radius').to(u.deg).value, -90)
        dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

        delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
        ra_min = (query_coords.ra.deg - delta_ra) % 360
        ra_max = (query_coords.ra.deg + delta_ra) % 360

        query = f"SELECT TOP {self.pipeline.max_records} * " \
            f"FROM dp1.Object WHERE coord_ra BETWEEN {ra_min} AND {ra_max} AND " \
            f" coord_dec BETWEEN {dec_min} AND {dec_max}"
        
        query = \
        """
        SELECT TOP {max_records} objectId, coord_ra, coord_dec, g_cModelMag, g_cModelMagErr, refExtendedness
        FROM dp1.Object
        WHERE coord_ra BETWEEN 4.0641 AND 106.8238
            AND coord_dec BETWEEN -72.7414 AND 8.0037
            AND g_cModelMag < 24 
            AND refExtendedness = 1
        """

        # Extended Chandra Deep Field South (ECDFS)
        query = \
        """
        SELECT TOP {max_records} objectId, coord_ra, coord_dec, g_cModelMag, g_cModelMagErr, refExtendedness
        FROM dp1.Object
        WHERE coord_ra BETWEEN 52 AND 54
            AND coord_dec BETWEEN -28 AND -26
        """


        query = query.format(max_records=self.pipeline.max_records)

        # sync
        table = client.query(query)

        # async
        # table = client.query_async(query)

        # convert table to pandas dataframe
        df = table.to_pandas()

        # remove rows with NaN values of 'g_cModelMag'
        df = df.dropna(subset=['g_cModelMag'])

        # convert back to table
        table = Table.from_pandas(df)

        self.output = df

        query_info = f"lsst_tap__limit{self.pipeline.max_records}__ra{ra_min:.4f}_{ra_max:.4f}__dec{dec_min:.4f}_{dec_max:.4f}"

        # first check cache
        if os.path.exists(f"{self.stage_dir}/{query_info}.csv"):
            print(f"File {self.stage_dir}/{query_info}.csv already exists. ")
            # first read the table
            existing_table = Table.read(f"{self.stage_dir}/{query_info}.csv", format="csv")
            existing_ids = set(existing_table['objectId'])
            mask = [oid not in existing_ids for oid in table['objectId']]
            new_rows = table[mask]
            existing_table = vstack([existing_table, new_rows])

            self.output = existing_table.to_pandas()

            existing_table.write(f"{self.stage_dir}/{query_info}.csv", format="csv", overwrite=True)

        else:
            table.write(f"{self.stage_dir}/{query_info}.csv", format="csv", overwrite=True)
            print(f"Saved query result to {self.stage_dir}/{query_info}.csv")


# ============================================================
# StageFetchSDSS_V2_ManualCutout
# ============================================================
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


# ============================================================
# StageFetchSDSS_V3_ManualCutout
# ============================================================
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
            print(f"Missing columns in DataFrame: {[col for col in required_columns if col not in df.columns]}")
            raise ValueError(f"DataFrame from previous stage must contain columns: {required_columns}. Actual columns: {df.columns.tolist()}")

        return True

    def run(self):
        # read the positions from the previous stage

        df, image_url_format_string = self.prev_stage.output

        print(image_url_format_string)

        astroosFetch = AstroosFetchManualFitsCutout(
            df=df,
            dir=self.stage_dir,
            image_url_format_string=image_url_format_string,
            dataset=self.dataset
        )

        output, labels, output_band_bounds = astroosFetch.fetch_images()

        self.output = output, labels, output_band_bounds

        images_tensor = torch.tensor(output, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        torch.save(images_tensor, f"{self.pipeline.dataset.dir}/X_train.pt")
        torch.save(labels_tensor, f"{self.pipeline.dataset.dir}/y_train.pt")
        print(f"Saved file: {self.pipeline.dataset.dir}/X_train.pt and {self.pipeline.dataset.dir}/y_train.pt")

        self.pipeline.X_train_filename = f"{self.pipeline.dataset.dir}/X_train.pt"
        self.pipeline.y_train_filename = f"{self.pipeline.dataset.dir}/y_train.pt"


# ============================================================
# StageFetchSDSS_V2_AutoCutout
# ============================================================
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


# ============================================================
# StageFetchLSSTSoda
# ============================================================
class StageFetchLSSTSoda(DataPipelineStage):
    """Data pipeline stage for fetching LSST data."""
    def __init__(self, dataset):
        super().__init__(stage_name="fetch", requires_stage_dir=True)
        self.dataset = dataset

    def _validate_prev_stage(self):
        return True

    def center_crop(self, x, crop_h, crop_w):
        _, _, h, w = x.shape
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return x[:, :, top:top+crop_h, left:left+crop_w]

    def run(self):
        
        # read the positions from the previous stage

        df = self.prev_stage.output

        print(f"Fetching LSST SODA cutout images for {len(df)} objects...")

        ra = df['coord_ra'].to_list()
        dec = df['coord_dec'].to_list()

        # list of tuples
        positions = list(zip(ra, dec))

        service = get_siav2_service("dp1")

        eff_wl = 622.1e-09
        time1 = Time(60623.256, format="mjd", scale="tai")
        time2 = Time(60623.259, format="mjd", scale="tai")

        table = None

        hdul_list = []

        for row in tqdm(df.itertuples(), total=len(df), desc="Downloading LSST SODA Cutout Images"):
            target_ra = row.coord_ra
            target_dec = row.coord_dec
            search_radius = 0.2
            circle = (target_ra, target_dec, search_radius)
            result = service.search(
                pos=circle,
                calib_level=2,
                dpsubtype='lsst.visit_image',
                band=eff_wl,
                time=(time1, time2),
            )
            print("Result:")
            print(result)

            if (len(result) > 0):

                datalink_url = result[0].access_url
                dl_result = DatalinkResults.from_result_url(datalink_url,
                                                            session=get_pyvo_auth())
                print(f"Datalink status: {dl_result.status}. Datalink service url: {datalink_url}")
                # continue

                try:

                    
                    sq = SodaQuery.from_resource(dl_result,
                                     dl_result.get_adhocservice_by_id("cutout-sync-exposure"),
                                     session=get_pyvo_auth())

                    print("sq: ", sq)

                    spherePoint = geom.SpherePoint(target_ra*geom.degrees, target_dec*geom.degrees)
                    Radius = search_radius * u.deg
                    sq.circle = (spherePoint.getRa().asDegrees() * u.deg,
                                spherePoint.getDec().asDegrees() * u.deg,
                                Radius)
                    
                    stream = sq.execute_stream()
                    
                    try:
                        cutout_bytes = stream.read()

                    except Exception as e:
                        print("Stream read failed.")
                        continue

                    
                    try:
                        # cutout_bytes is a FITS file in bytes
                        hdul = fits.open(io.BytesIO(cutout_bytes))
                        print(hdul.info())

                        hdul_list.append(hdul)


                        # arr = hdul[1].data
                        # if not arr.dtype.isnative:
                        #     arr = arr.view(arr.dtype.newbyteorder('=')) 
                        # cropped_arr = self.center_crop(arr, 100, 100)
                        # tensor = torch.from_numpy(cropped_arr)
                        # print(tensor.shape)
                    except Exception as e:
                        print("no valid hdul", e)



                    if table is None:
                        table = result.to_table()
                    else:
                        table = vstack([table, result.to_table()], join_type='outer', metadata_conflicts='silent')

                    print(f"Fetched {len(result)} images for objectId {row.objectId} at RA: {target_ra}, Dec: {target_dec}")
                    print(result.to_table())
                except Exception as e:
                    print("an error has occured:", e)

        if table is not None:
            print(f"Downloaded {len(table)} LSST SODA cutout images.")
            print(table)

        nchw = torch.rand((len(tensors), 1, 100, 100))
        for tensor in tensors:
            nchw = torch.cat((nchw, tensor.unsqueeze(0)), dim=0)


        print("nchw shape:", nchw.shape)
        torch.save(nchw, f"{self.pipeline.dataset.dir}/X_train.pt")
        print(f"Saved file: {self.pipeline.dataset.dir}/X_train.pt")

# ============================================================
# StageFilterCatalogSDSS
# ============================================================
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



# ============================================================
# StageFilterCatalogLSST
# ============================================================
class StageFilterCatalogLSST(DataPipelineStage):
    """
    Data pipeline stage for filtering SDSS catalog results.
    """

    def __init__(self, pipeline_dir, stage_dir, pipeline):
        super().__init__(stage_name=stage_dir, pipeline_dir=pipeline_dir, stage_dir=stage_dir, pipeline=pipeline)

    def run(self):

        print("Filtering SDSS catalog results...")

        

        catalog = AstroosCatalogLSST(dir=self.pipeline_dir, 
                                          catalog_dir=self.prev_stage_dir, 
                                          pipeline=self.pipeline, 
                                          max_records=self.pipeline.max_records)

        catalog.load_single_catalog()
        pos, labels, rows = catalog.filter_catalog()
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


# ============================================================
# StageFetchSDSS
# ============================================================
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


# ============================================================
# StageFetchLSST
# ============================================================
class StageFetchLSST(DataPipelineStage):
    """
    Data pipeline stage for fetching LSST images.
    """

    def __init__(self, pipeline_dir, stage_dir, pipeline, chunk_size=100):
        super().__init__(stage_name=stage_dir, pipeline_dir=pipeline_dir, stage_dir=stage_dir, pipeline=pipeline)
        self.chunk_size = chunk_size

    def run(self):

        print("Fetching LSST images...")

        positions = self.pipeline.filtered_positions
        labels_list = self.pipeline.filtered_labels
        rows_list = self.pipeline.filtered_rows

        print(f"Total positions to fetch: {len(positions)} total labels: {len(labels_list)}")
        print(positions)
        print(labels_list)

        astroos_fetch = AstroosFetchLSST(label_definitions=self.pipeline.label_definitions)


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
