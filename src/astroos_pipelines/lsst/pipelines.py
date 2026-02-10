
import sys
import numpy as np
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy import units as u

import importlib

from utils.formatting_utils import ascii_kv_table
importlib.reload(sys.modules['utils.formatting_utils'])

from astroos_pipelines.query import AstroosQueryLSST

from astroos_pipelines.pipelines import DataPipelineStage
from utils.rsp_utils import get_cutout_bands

importlib.reload(sys.modules['utils.rsp_utils'])
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

from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
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

        # sync
        table = client.query(query)

        # async
        # table = client.query_async(query)

        # convert table to pandas dataframe
        df = table.to_pandas()

        # add label from Simbad crossmatch if available
        for i, row in df.iterrows():

            query = \
            """
            SELECT TOP 1 * 
            FROM basic b JOIN otypedef o ON b.otype = o.otype 
            -- WHERE main_id LIKE 'SDSS%' AND
            WHERE 
            ra >= {ra_min}
            AND ra < {ra_max} 
            AND dec >= {dec_min} 
            AND dec <= {dec_max} 
            AND (o.otype_longname = 'Galaxy' OR o.otype_longname = 'Star')
            -- b.rvz_redshift < 0.05 AND
            -- (b.morph_type IS NOT NULL)

            ;
            """
            query = query.format(
                ra_min=row['coord_ra'] - 0.01, 
                ra_max=row['coord_ra'] + 0.01, 
                dec_min=row['coord_dec'] - 0.01, 
                dec_max=row['coord_dec'] + 0.01,
            )
            res = Simbad.query_tap(query)
            # print(f"Query: {query}, number of results: {len(res) if res is not None else 0}")

            label_index = -1  # default to -1 for unknown
            for match_data in res:
                pass
                #for i in match_data.colnames:
                    #print(f"{i}, {match_data[i]}")
                morph_type = str(match_data['morph_type'])
                print(f"Simbad data found for {match_data['main_id']}. Type: {match_data['otype_longname']}, Morphological type: [{morph_type}]")


                label_index = self.pipeline.dataset.labels._get_label_index(morph_type)
            
            # print("Simbad: ")
            # print(res)
            df.at[i, 'label'] = label_index

        # convert back to table
        table = Table.from_pandas(df)

        self.output = df

        query_info = f"lsst_tap__limit{self.pipeline.max_records}__ra{ra_min:.4f}_{ra_max:.4f}__dec{dec_min:.4f}_{dec_max:.4f}"

        # first check cache
        # do this later
        # if os.path.exists(f"{self.stage_dir}/{query_info}.csv"):
            # log.info(f"File {self.stage_dir}/{query_info}.csv already exists. ")
            # # first read the table
            # existing_table = Table.read(f"{self.stage_dir}/{query_info}.csv", format="csv")
            # existing_ids = set(existing_table['objectId'])
            # mask = [oid not in existing_ids for oid in table['objectId']]
            # new_rows = table[mask]
            # existing_table = vstack([existing_table, new_rows])

            # self.output = existing_table.to_pandas()

            # existing_table.write(f"{self.stage_dir}/{query_info}.csv", format="csv", overwrite=True)

        # else:
        table.write(f"{self.stage_dir}/{query_info}.csv", format="csv", overwrite=True)
        log.info(f"Saved query result to {self.stage_dir}/{query_info}.csv")



# ============================================================
# StageFetchLSSTSoda
# ============================================================
class StageFetchLSSTSoda(DataPipelineStage):
    """
    Data pipeline stage for fetching LSST data via SIA and SODA.
    """
    def __init__(self, dataset):
        super().__init__(stage_name="fetch", requires_stage_dir=True)
        self.dataset = dataset

    def _validate_prev_stage(self):
        return rsp_mode

    def run(self):

        # read the positions from the previous stage
        df = self.prev_stage.output
        n = len(df)
        print(f"Fetching LSST SODA cutout images for {n} objects...")

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

            hdu_phot = fits.ImageHDU(data=photometric_features, name="PHOTO")
            hdu_phot.header['label'] = int(row.label) if hasattr(row, "label") else 0
            hdu_phot.header['ra'] = float(target_ra)
            hdu_phot.header['dec'] = float(target_dec)
            hdu_phot.header['main_id'] = int(row.objectId)

            hdul.append(hdu_img)
            hdul.append(hdu_phot)

            self.dataset.append(hdul)

            

        # print("nchw shape:", nchw.shape)
        # torch.save(nchw, f"{self.pipeline.dataset.dir}/X_train.pt")
        # print(f"Saved file: {self.pipeline.dataset.dir}/X_train.pt")
        # # placeholder labels
        # labels_tensor = torch.zeros((n,), dtype=torch.int64)
        # torch.save(labels_tensor, f"{self.pipeline.dataset.dir}/y_train.pt")
        # print(f"Saved file: {self.pipeline.dataset.dir}/y_train.pt")
