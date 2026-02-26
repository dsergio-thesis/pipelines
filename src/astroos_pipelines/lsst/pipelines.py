
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import sys
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astropy import units as u

import importlib

from astroos_pipelines.lsst.query import AstroosQueryLSST
from astroos_pipelines.pipelines import DataPipelineStage
from astroos_pipelines.utils.rsp import get_cutout_bands

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.rsp'])
importlib.reload(sys.modules['astroos_pipelines.query'])

from astropy.io import fits
# do wcs next

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    from lsst.rsp.utils import get_pyvo_auth
    import lsst.geom as geom
    from lsst.afw.fits import MemFileManager

    # other LSST dependencies
    from pyvo.dal.adhoc import DatalinkResults
    from astropy.time import Time
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery
    rsp_mode = True

except ImportError as e:
    print(f"LSST RSP dependencies not found. RSP mode will be disabled. Please install the required packages: {e}")
    pass

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class StageCatalogLSST(DataPipelineStage):
    """
    Data pipeline stage for cataloging LSST data via TAP.
    """

    def __init__(self):
        super().__init__(stage_name="catalog", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return rsp_mode
    
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


        # Extended Chandra Deep Field South (ECDFS)
        query = \
        """
        SELECT TOP {max_records}
        objectId,
        tract,
        patch,
        coord_ra,
        coord_dec,

        -- u
        u_psfFlux,            u_psfFluxErr,            u_psfFlux_flag,
        u_free_cModelFlux,    u_free_cModelFluxErr,    u_free_cModelFlux_flag,

        -- g
        g_psfFlux,            g_psfFluxErr,            g_psfFlux_flag,
        g_free_cModelFlux,    g_free_cModelFluxErr,    g_free_cModelFlux_flag,

        -- r
        r_psfFlux,            r_psfFluxErr,            r_psfFlux_flag,
        r_free_cModelFlux,    r_free_cModelFluxErr,    r_free_cModelFlux_flag,

        -- i
        i_psfFlux,            i_psfFluxErr,            i_psfFlux_flag,
        i_free_cModelFlux,    i_free_cModelFluxErr,    i_free_cModelFlux_flag,

        -- z
        z_psfFlux,            z_psfFluxErr,            z_psfFlux_flag,
        z_free_cModelFlux,    z_free_cModelFluxErr,    z_free_cModelFlux_flag,

        -- y
        y_psfFlux,            y_psfFluxErr,            y_psfFlux_flag,
        y_free_cModelFlux,    y_free_cModelFluxErr,    y_free_cModelFlux_flag,

        refExtendedness

        FROM dp1.Object
        -- WHERE coord_ra BETWEEN 52 AND 54
        --   AND coord_dec BETWEEN -28 AND -26

        -- WHERE coord_ra BETWEEN 53.1 AND 53.2
        --  AND coord_dec BETWEEN -27.9 AND -27.6
        WHERE coord_ra BETWEEN {ra_min} AND {ra_max}
            AND coord_dec BETWEEN {dec_min} AND {dec_max}

        -- AND objectId = 611255072642319851
        """

        query = query.format(
                max_records=self.pipeline.max_records,
                ra_min=ra_min,
                ra_max=ra_max,
                dec_min=dec_min,
                dec_max=dec_max
                )

        # print(f"Query: {query}")
        
        # sync
        table = client.query(query)

        col_names = table.colnames
        col_types = [str(table[name].dtype) for name in col_names]
        print("Query result columns and types:")


        # async
        # table = client.query_async(query)

        self.output = table
        self.cache_pipeline_output()
        print(f"number of results: {len(self.output)}")
        print(self.output)


# ============================================================
# StageMatchLSSTtoHST 
# ============================================================
class StageMatchLSSTtoHST(DataPipelineStage): 
    """
    Data pipeline stage for cross-matching LSST catalog with HST labels.
    """
    def __init__(self):
        super().__init__(stage_name="match", requires_stage_dir=True)

    def _validate_prev_stage(self):
        if not rsp_mode:
            log.error("RSP mode is not available. Cannot run StageMatchLSSTtoHST.")
            return False

        required_columns = {'objectId', 'coord_ra', 'coord_dec'}
        if not all(col in self.prev_stage.output.columns for col in required_columns):
            log.error(f"Previous stage output is missing required columns: {required_columns}")
            return False
        return True

    def run(self):

        # read the table from the previous stage 
        table = self.prev_stage.output

        self.output = Table.from_pandas(AstroosQueryLSST.cross_match_labels_hst(table.to_pandas(), "catalogs/hst/hst.fits"))

        # print("pipeline labels match: ")
        # print(self.output['label'].value_counts())

        self.cache_pipeline_output()

# ============================================================
# StagePreprocessLSST
# ============================================================
class StagePreprocessLSST(DataPipelineStage):
    """
    Data pipeline stage for preprocessing LSST catalog features.
    Main purpose is to transform the raw fluxes and errors into a more ML-friendly format, and to store them in the dataset for later use.
    """
    def __init__(self):
        super().__init__(stage_name="preprocess", requires_stage_dir=True)
    def _validate_prev_stage(self):
        if not rsp_mode:
            log.error("RSP mode is not available. Cannot run StagePreprocessLSST.")
            return False
        required_columns = {'objectId', 'coord_ra', 'coord_dec'}
        if not all(col in self.prev_stage.output.columns for col in required_columns):
            log.error(f"Previous stage output is missing required columns: {required_columns}")
            return False
        return True

    def run(self):

        df = self.prev_stage.output.to_pandas()
        n = len(df)
        print(f"Feature preprocesing for {n} objects...")

        bands = ['u', 'g', 'r', 'i', 'z']  # add 'y' 
        num_bands = len(bands)

        # precompute safe scales (dataset-level)
        flux_scale = self.median_r_psfFlux if getattr(self, "median_r_psfFlux", 0) and self.median_r_psfFlux > 0 else 1.0
        err_scale  = self.median_r_psfFluxErr if getattr(self, "median_r_psfFluxErr", 0) and self.median_r_psfFluxErr > 0 else 1.0

        for row in tqdm(df.itertuples(), total=n, desc="Extracting Photometric Features"):
            target_ra = row.coord_ra
            target_dec = row.coord_dec

            # 4 features per band: transformed flux, transformed err, log SNR, bad-flag
            photometric_features = np.zeros((num_bands, 4), dtype=np.float32)

            for bi, band in enumerate(bands):
                flux = getattr(row, f"{band}_psfFlux", None)
                err  = getattr(row, f"{band}_psfFluxErr", None)
                flag = getattr(row, f"{band}_psfFlux_flag", False)

                # sanitize missing/NaN
                if flux is None or err is None or (isinstance(flux, float) and np.isnan(flux)) or (isinstance(err, float) and np.isnan(err)):
                    x1 = 0.0
                    x2 = 0.0
                    x3 = 0.0
                    bad = 1.0  # treat missing as bad
                else:
                    # arcsinh scaling 
                    x1 = np.arcsinh(float(flux) / flux_scale)
                    x2 = np.arcsinh(float(err) / err_scale)

                    # SNR feature (clamp to non-negative)
                    if err > 0:
                        snr = float(flux) / float(err)
                        x3 = np.log1p(max(0.0, snr))
                    else:
                        x3 = 0.0

                    bad = 1.0 if bool(flag) else 0.0

                photometric_features[bi] = (x1, x2, x3, bad)

            hdu_phot = fits.ImageHDU(data=photometric_features, name="PHOTO")
            hdu_phot.header['label'] = int(row.label) if hasattr(row, "label") else 0
            hdu_phot.header['ra'] = float(target_ra)
            hdu_phot.header['dec'] = float(target_dec)
            hdu_phot.header['objectId'] = int(row.objectId)

            dataset = self.pipeline.dataset

            if (dataset.contains(row.objectId)):
                dataset.update(row.objectId, hdu_phot)
            else:
                dataset.append(hdu_phot)

        self.output = Table.from_pandas(df)


# ============================================================
# StageFetchLSSTSoda
# ============================================================
class StageFetchLSSTSoda(DataPipelineStage):
    """
    Data pipeline stage for fetching LSST data via SIA and SODA.
    """
    def __init__(self):
        super().__init__(stage_name="fetch", requires_stage_dir=True)

    def _validate_prev_stage(self):
        if not rsp_mode:
            log.error("RSP mode is not available. Cannot run StageFetchLSSTSoda.")
            return False

        required_columns = {'objectId', 'coord_ra', 'coord_dec'}
        if not all(col in self.prev_stage.output.columns for col in required_columns):
            log.error(f"Previous stage output is missing required columns: {required_columns}")
            return False
        return True

    def run(self):

        # read the positions from the previous stage
        df = self.prev_stage.output.to_pandas()
        n = len(df)
        print(f"Fetching LSST SODA cutout images for {n} objects...")

        bands = ['u', 'g', 'r', 'i', 'z']  # add 'y' 
        num_bands = len(bands)

        # for row in tqdm(df.itertuples(), total=n, desc="Downloading LSST SODA Cutout Images"):
            # target_ra = row.coord_ra
            # target_dec = row.coord_dec

            # # Get cutouts (num_bands, 200, 200)
            # band_images = get_cutout_bands(
                # target_ra=target_ra,
                # target_dec=target_dec,
                # bands=bands
            # )

            # hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
            # hdu_img.header['label'] = int(row.label) if hasattr(row, "label") else 0
            # hdu_img.header['ra'] = float(target_ra)
            # hdu_img.header['dec'] = float(target_dec)
            # hdu_img.header['objectId'] = int(row.objectId)
            # hdu_img.header['rvz_redshift'] = -999
            # hdu_img.header['min_ra'] = float(target_ra - 0.0138889)
            # hdu_img.header['max_ra'] = float(target_ra + 0.0138889)
            # hdu_img.header['min_dec'] = float(target_dec - 0.0138889)
            # hdu_img.header['max_dec'] = float(target_dec + 0.0138889)

            # dataset = self.pipeline.dataset

            # if (dataset.contains(row.objectId)):
                # # print(f"dataset contains {row.objectId}")
                # dataset.update(row.objectId, hdu_img)
            # else:
                # # print(f"dataset DOES NOT contain {row.objectId}")
                # dataset.append(hdu_img)

        # self.output = Table.from_pandas(df)




        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_row(row):
            target_ra = row.coord_ra
            target_dec = row.coord_dec

            band_images = get_cutout_bands(
                target_ra=target_ra,
                target_dec=target_dec,
                bands=bands
            )

            return row, band_images

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_row, row)
                       for row in df.itertuples()]

            for future in tqdm(as_completed(futures), total=len(futures)):
                row, band_images = future.result()
                # build HDU here

                target_ra = row.coord_ra
                target_dec = row.coord_dec

                hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
                hdu_img.header['label'] = int(row.label) if hasattr(row, "label") else 0
                hdu_img.header['ra'] = float(target_ra)
                hdu_img.header['dec'] = float(target_dec)
                hdu_img.header['objectId'] = int(row.objectId)
                hdu_img.header['rvz_redshift'] = -999
                hdu_img.header['min_ra'] = float(target_ra - 0.0138889)
                hdu_img.header['max_ra'] = float(target_ra + 0.0138889)
                hdu_img.header['min_dec'] = float(target_dec - 0.0138889)
                hdu_img.header['max_dec'] = float(target_dec + 0.0138889)

                dataset = self.pipeline.dataset

                if (dataset.contains(row.objectId)):
                    # print(f"dataset contains {row.objectId}")
                    dataset.update(row.objectId, hdu_img)
                else:
                    # print(f"dataset DOES NOT contain {row.objectId}")
                    dataset.append(hdu_img)
        self.output = Table.from_pandas(df)



class StageButlerFetchLSST(DataPipelineStage):
    """
    Data pipeline stage for fetching LSST data via the Butler.
    """
    def __init__(self):
        super().__init__(stage_name="fetch_butler", requires_stage_dir=True)

    def _validate_prev_stage(self):
        if not rsp_mode:
            log.error("RSP mode is not available. Cannot run StageButlerFetchLSST.")
            return False

        required_columns = {'objectId', 'coord_ra', 'coord_dec'}
        if not all(col in self.prev_stage.output.columns for col in required_columns):
            log.error(f"Previous stage output is missing required columns: {required_columns}")
            return False
        return True

    def run(self):

        # read the positions from the previous stage
        df = self.prev_stage.output.to_pandas()
        n = len(df)
        print(f"Fetching LSST data via Butler for {n} objects...")

        objects = self.prev_stage.output
        
        import pandas as pd

        rsp_mode = False
        try:
            from lsst.rsp import get_tap_service
            import lsst.geom as geom
            rsp_mode = True
        except ImportError:
            pass




        tasks = build_groups(objects, self.pipeline.dataset)

        with ProcessPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(worker_patch, tasks):
                pass






def worker_patch(args):

    BANDS = ["g", "r", "i", "z", "y"]
    BANDS = ["g", "r", "i", "z"]
    # cutout stamp size (pixels)
    STAMP_W = 100
    STAMP_H = 100

    tract, patch, object_rows, dataset = args

    from lsst.daf.butler import Butler
    butler = Butler("dp1", collections="LSSTComCam/DP1")

    # Load each band ONCE per patch
    coadds = {
        b: butler.get("deep_coadd", tract=tract, patch=patch, band=b)
        for b in BANDS
    }

    ext = geom.Extent2I(STAMP_W, STAMP_H)
    band_images = np.zeros((len(BANDS), STAMP_H, STAMP_W), dtype=np.float32)

    for row in object_rows:
        ra_deg = float(row["coord_ra"])
        dec_deg = float(row["coord_dec"])

        # cross-match with HST


        # SpherePoint expects (lon, lat) as Angles.
        # Use degrees explicitly.
        sky = geom.SpherePoint(ra_deg * geom.degrees, dec_deg * geom.degrees)

        for band, exp in coadds.items():
            wcs = exp.getWcs()
            if wcs is None:
                # Shouldn't happen for coadds, but guard anyway
                continue

            # Convert sky coordinate to pixel coordinate in this exposure
            pix = wcs.skyToPixel(sky)  # returns lsst.geom.Point2D

            # Optional: skip objects whose pixel center is off-image
            bbox = exp.getBBox()
            if not bbox.contains(geom.Point2I(int(round(pix.getX())), int(round(pix.getY())))):
                continue

            cutout = exp.getCutout(pix, ext)
            band_images[BANDS.index(band)] = cutout.getImage().getArray()

        target_ra = ra_deg 
        target_dec = dec_deg

        hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
        hdu_img.header['label'] = int(row.label) if hasattr(row, "label") else 0
        hdu_img.header['ra'] = float(target_ra)
        hdu_img.header['dec'] = float(target_dec)
        hdu_img.header['objectId'] = int(row['objectId'])
        hdu_img.header['rvz_redshift'] = -999
        hdu_img.header['min_ra'] = float(target_ra - 0.0138889)
        hdu_img.header['max_ra'] = float(target_ra + 0.0138889)
        hdu_img.header['min_dec'] = float(target_dec - 0.0138889)
        hdu_img.header['max_dec'] = float(target_dec + 0.0138889)

        # dataset = self.pipeline.dataset

        if (dataset.contains(row.objectId)):
            # print(f"dataset contains {row.objectId}")
            dataset.update(row.objectId, hdu_img)
        else:
            # print(f"dataset DOES NOT contain {row.objectId}")
            dataset.append(hdu_img)

    return len(object_rows)

def build_groups(objects, dataset):
    groups = defaultdict(list)
    for row in objects:
        groups[(int(row["tract"]), int(row["patch"]))].append(row)
    return [(t, p, rows, dataset) for (t, p), rows in groups.items()]
