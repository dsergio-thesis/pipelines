
from os import wait
import sys
import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy import units as u

import importlib

from astroos_pipelines.utils.formatting_utils import ascii_kv_table
from astroos_pipelines.lsst.query import AstroosQueryLSST
from astroos_pipelines.pipelines import DataPipelineStage
from astroos_pipelines.utils.rsp_utils import get_cutout_bands

importlib.reload(sys.modules['astroos_pipelines.utils.formatting_utils'])
importlib.reload(sys.modules['astroos_pipelines.utils.rsp_utils'])
importlib.reload(sys.modules['astroos_pipelines.query'])

from astropy.io import fits
from astropy.wcs import WCS

from astroquery.simbad import Simbad
from astroquery.sdss import SDSS

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


        # async
        # table = client.query_async(query)

        self.output = table.to_pandas()
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

        # read the table from the previous stage df = self.prev_stage.output

        self.output = AstroosQueryLSST.cross_match_labels_hst(df, "catalogs/hst/hst.fits")

        print("pipeline labels match: ")
        print(self.output['label'].value_counts())

        cache_pipeline_output()

# ============================================================
# StagePreprocessLSST
# ============================================================
class StagePreprocessLSST(DataPipelineStage):
    """
    Data pipeline stage for preprocessing LSST catalog features.
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

        df = self.prev_stage.output
        n = len(df)
        print(f"Feature preprocesing for {n} objects...")

        bands = ['u', 'g', 'r', 'i', 'z']  # add 'y' 
        num_bands = len(bands)

        # precompute safe scales (dataset-level)
        flux_scale = self.median_r_psfFlux if getattr(self, "median_r_psfFlux", 0) and self.median_r_psfFlux > 0 else 1.0
        err_scale  = self.median_r_psfFluxErr if getattr(self, "median_r_psfFluxErr", 0) and self.median_r_psfFluxErr > 0 else 1.0

        for row in tqdm(df.itertuples(), total=n, desc="Downloading LSST SODA Cutout Images"):
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
            hdu_phot.header['main_id'] = int(row.objectId)

            hdul.append(hdu_phot)

            dataset = self.pipeline.dataset

            if (dataset.contains(row.objectId)):
                # update existing entry
                existing_hdul = dataset.get(row.objectId)
                existing_hdul["PHOTO"].data = photometric_features 
                dataset.update(row.objectId, existing_hdul)
            else:
                dataset.append(hdul)

        self.output = df


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
        df = self.prev_stage.output
        n = len(df)
        print(f"Fetching LSST SODA cutout images for {n} objects...")

        bands = ['u', 'g', 'r', 'i', 'z']  # add 'y' 
        num_bands = len(bands)

        for row in tqdm(df.itertuples(), total=n, desc="Downloading LSST SODA Cutout Images"):
            target_ra = row.coord_ra
            target_dec = row.coord_dec

            # Get cutouts (num_bands, 200, 200)
            band_images = get_cutout_bands(
                target_ra=target_ra,
                target_dec=target_dec,
                bands=bands
            )

            # Build FITS container
            hdul = fits.HDUList([fits.PrimaryHDU()])

            hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
            hdu_img.header['label'] = int(row.label) if hasattr(row, "label") else 0
            hdu_img.header['ra'] = float(target_ra)
            hdu_img.header['dec'] = float(target_dec)
            hdu_img.header['main_id'] = int(row.objectId)
            hdu_img.header['rvz_redshift'] = -999
            hdu_img.header['min_ra'] = float(target_ra - 0.0138889)
            hdu_img.header['max_ra'] = float(target_ra + 0.0138889)
            hdu_img.header['min_dec'] = float(target_dec - 0.0138889)
            hdu_img.header['max_dec'] = float(target_dec + 0.0138889)

            hdul.append(hdu_img)

            dataset = self.pipeline.dataset

            if (dataset.contains(row.objectId)):
                # update existing entry
                existing_hdul = dataset.get(row.objectId)
                existing_hdul["CUTOUTS"].data = band_images
                dataset.update(row.objectId, existing_hdul)
            else:
                dataset.append(hdul)

        self.output = df


