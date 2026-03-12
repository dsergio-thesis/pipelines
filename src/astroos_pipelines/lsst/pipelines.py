
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import sys
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astropy import units as u
import pandas as pd
import importlib

from astroos_pipelines.lsst.query import AstroosQueryLSST
from astroos_pipelines.pipelines import StagePipeline 
from astroos_pipelines.utils.rsp import get_cutout_bands
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.rsp'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])

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


class StageCatalogLSST(StagePipeline):
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

        detect_fromBlend, detect_isIsolated,

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
# StageLSSTExploratoryDataAnalysis
# ============================================================
class StageLSSTExploratoryDataAnalysis(StagePipeline):
    """
    Data pipeline stage for exploratory data analysis of LSST catalog.
    """
    def __init__(self):
        super().__init__(stage_name="eda", requires_stage_dir=True)

    def _validate_prev_stage(self):
        if not rsp_mode:
            log.error("RSP mode is not available. Cannot run StageLSSTExploratoryDataAnalysis.")
            return False

        required_columns = {'objectId', 'coord_ra', 'coord_dec'}
        if not all(col in self.prev_stage.output.columns for col in required_columns):
            log.error(f"Previous stage output is missing required columns: {required_columns}")
            return False
        return True

    def run(self):

        # read the table from the previous stage 
        table = self.prev_stage.output

        print("Basic statistics of the catalog:")
        print(table.info)
        print(table.describe())

        # plot distributions of key features
        import matplotlib.pyplot as plt
        import seaborn as sns
        df = table.to_pandas()
        plt.figure(figsize=(10, 6))
        sns.histplot(df['r_psfFlux'], bins=50, kde=True)
        plt.title("Distribution of r-band PSF Flux")
        plt.xlabel("r_psfFlux")
        plt.ylabel("Count")
        plt.savefig(self.stage_dir / "r_psfFlux_distribution.png")
        plt.close()



# ============================================================
# StageMatchLSSTtoHST 
# ============================================================
class StageMatchLSSTtoHST(StagePipeline): 
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
class StagePreprocessLSST(StagePipeline):
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
        bands = ['u', 'g', 'r', 'i', 'z', 'y']

        label_counts = dict()

        num_bands = len(bands)

        # precompute safe scales (dataset-level)
        flux_scale = self.median_r_psfFlux if getattr(self, "median_r_psfFlux", 0) and self.median_r_psfFlux > 0 else 1.0
        err_scale  = self.median_r_psfFluxErr if getattr(self, "median_r_psfFluxErr", 0) and self.median_r_psfFluxErr > 0 else 1.0

        for row in tqdm(df.itertuples(), total=n, desc="Extracting Photometric Features"):
            target_ra = row.coord_ra
            target_dec = row.coord_dec

            if hasattr(row, "label"):
                if (str(row.label) in label_counts):
                    # print(f"found label {str(row.label)}, adding to existing counts")
                    label_counts[str(row.label)] += 1
                else:
                    # print(f"found label {str(row.label)}, setting count to 1")
                    label_counts[str(row.label)] = 1

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

        print(f"Label counts: {label_counts}")

        self.output = Table.from_pandas(df)


# ============================================================
# StageFetchLSSTSoda
# ============================================================
class StageFetchLSSTSoda(StagePipeline):
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
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
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
                hdu_img.header['redshift'] = -999
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



class StageButlerFetchLSST(StagePipeline):
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
        objects = self.prev_stage.output


        print(f"Fetching LSST data via Butler for {len(objects)} objects...")
        # print(objects['label'])

        tasks = build_groups(
                objects, 
                self.pipeline.dataset.get_dataset_dir(),
                self.pipeline.dataset.get_labels_file()
                )

        with ProcessPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(worker_patch, tasks):
                pass



def worker_patch(args):

    BANDS = ["u", "g", "r", "i", "z"]
    BANDS = ["u", "g", "r", "i", "z", "y"]

    # cutout stamp size (pixels)
    STAMP_W = 100
    STAMP_H = 100

    tract, patch, object_rows, dataset_dir, dataset_labels = args

    # print("object_rows")
    # print(object_rows)

    dataset = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=dataset_dir,
            labels_init_file=dataset_labels,
            N_bands=len(BANDS), 
            N_morphometric_features=4,
            N_photometric_features=4,
            )

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

        # print("row\n\n")
        # print(row)
        ra_deg = float(row["coord_ra"])
        dec_deg = float(row["coord_dec"])

        # SpherePoint expects (lon, lat) as Angles.
        # Use degrees explicitly.
        sky = geom.SpherePoint(ra_deg * geom.degrees, dec_deg * geom.degrees)

        wcs_header = fits.Header()
        hdr = fits.Header()
        
        min_ra = ra_deg - 0.0138889
        max_ra = ra_deg + 0.0138889
        min_dec = dec_deg - 0.0138889
        max_dec = dec_deg + 0.0138889

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
            # get minimal WCS info for the cutout
            wcs_cutout = cutout.getWcs()


            bbox = cutout.getBBox()
            # hdr = fits_header_from_lsst_cutout_wcs(cutout.getWcs(), bbox)
            hdr = make_cutout_header3(cutout, ra_deg, dec_deg)
            # print(f"hdr: {hdr}")


            # print("cutout bbox:", cutout.getBBox())  # should show a small region
            # print("cutout dims:", cutout.getDimensions())  # width/height
            # print("cutout WCS:", wcs_cutout)  # should be a valid WCS object

            if (band == "r"):
                wcs_header = wcs_cutout.getFitsMetadata()

                # min_ra, max_ra = wcs_cutout.getSkyBBox().getMin().getX(), wcs_cutout.getSkyBBox().getMax().getX()
                # min_dec, max_dec = wcs_cutout.getSkyBBox().getMin().getY(), wcs_cutout.getSkyBBox().getMax().getY()
                # min_ra, max_ra, min_dec, max_dec = wcs_bounds_radec(wcs_cutout, STAMP_W, STAMP_H)
            
            band_images[BANDS.index(band)] = cutout.getImage().getArray()

        target_ra = ra_deg 
        target_dec = dec_deg

        
        

        # print(f"band_images shape: {band_images.shape}")

        hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
        hdu_img.header['label'] = int(row['label'])
        hdu_img.header['ra'] = float(target_ra)
        hdu_img.header['dec'] = float(target_dec)
        hdu_img.header['objectId'] = int(row['objectId'])
        hdu_img.header['redshift'] = -999
        # hdu_img.header['min_ra'] = min_ra
        # hdu_img.header['max_ra'] = max_ra
        # hdu_img.header['min_dec'] = min_dec
        # hdu_img.header['max_dec'] = max_dec

        for k, v in hdr.items():
            hdu_img.header[k] = v
            # print(f"wcs header: {k}: {v}")

        if (dataset.contains(row['objectId'])):
            # print(f"dataset contains {row['objectId']}")
            dataset.update(row['objectId'], hdu_img)
        else:
            # print(f"dataset DOES NOT contain {row['objectId']}")
            dataset.append(hdu_img)

    return len(object_rows)

def build_groups(objects, dataset_dir, labels_init_file):
    groups = defaultdict(list)
    for row in objects:
        groups[(int(row["tract"]), int(row["patch"]))].append(row)
    return [(t, p, rows, dataset_dir, labels_init_file) for (t, p), rows in groups.items()]



import numpy as np
import lsst.geom as geom

def wcs_bounds_radec(skywcs, width: int, height: int):
    """
    Return (ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg) for an image
    with given width/height in pixels using an LSST SkyWcs.

    Handles RA wrap-around (e.g., near 0/360).
    """
    # LSST pixel coords are (x, y). Use edge pixels.
    corners_pix = [
        geom.Point2D(0, 0),
        geom.Point2D(width - 1, 0),
        geom.Point2D(0, height - 1),
        geom.Point2D(width - 1, height - 1),
    ]

    ras = []
    decs = []
    for p in corners_pix:
        sp = skywcs.pixelToSky(p)  # returns lsst.geom.SpherePoint
        ras.append(sp.getRa().asDegrees())
        decs.append(sp.getDec().asDegrees())

    ras = np.array(ras, dtype=float)
    decs = np.array(decs, dtype=float)

    # Fix RA wrap: if corners straddle 0°, unwrap so min/max make sense
    ras_rad = np.deg2rad(ras)
    ras_unwrapped_deg = np.rad2deg(np.unwrap(ras_rad))

    ra_min = float(ras_unwrapped_deg.min())
    ra_max = float(ras_unwrapped_deg.max())

    # put back into [0, 360)
    ra_min = ra_min % 360.0
    ra_max = ra_max % 360.0

    dec_min = float(decs.min())
    dec_max = float(decs.max())

    return ra_min, ra_max, dec_min, dec_max


from astropy.io import fits

import lsst.geom as geom
from astropy.io import fits

def make_cutout_header3(cutout, ra, dec):
    wcs = cutout.getWcs()

    # Start from LSST's FITS WCS keywords
    hdr = fits.Header(wcs.getFitsMetadata().toDict())
    hdr["WCSAXES"] = 2

    # Choose a nice reference pixel in CUTOUT coordinates
    width, height = cutout.getDimensions()  # (W, H)
    x_ref = (width - 1) / 2.0
    y_ref = (height - 1) / 2.0

    # Find the sky coord at that cutout pixel
    sp = wcs.pixelToSky(geom.Point2D(x_ref, y_ref))
    ra_ref = sp.getRa().asDegrees()
    dec_ref = sp.getDec().asDegrees()

    # FITS uses 1-based pixel coordinates for CRPIX
    hdr["CRPIX1"] = x_ref + 1.0
    hdr["CRPIX2"] = y_ref + 1.0
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec

    ra_min, ra_max, dec_min, dec_max = wcs_bounds_radec(wcs, width, height)

    ra_min = ra + hdr['CD1_1'] * width / 2
    ra_max = ra - hdr['CD1_1'] * width / 2
    dec_min = dec - hdr['CD2_2'] * height / 2
    dec_max = dec + hdr['CD2_2'] * height / 2

    hdr["min_ra"] = ra_min
    hdr["max_ra"] = ra_max
    hdr["min_dec"] = dec_min
    hdr["max_dec"] = dec_max

    return hdr
