import matplotlib.gridspec as gridspec

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
# from astroos_pipelines.pipelines import StagePipeline 
from astroos_pipelines.dag import *
from astroos_pipelines.utils.rsp import get_cutout_bands
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.utils.plots.dataset_eda import dataset_eda

# importlib.reload(sys.modules['astroos_pipelines.dag'])
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.rsp'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])

import matplotlib.pyplot as plt
import seaborn as sns

from astropy.io import fits

import lsst.geom as geom
from astropy.io import fits
import numpy as np
import lsst.geom as geom


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


class NodeTAPQuery(Node):
    """
    A node that connects to a TAP service.

    """

    def __init__(self,
                 dag_dir=None,
                 node_type="NodeTAPQuery",
                 node_id=None,
                 parents=[],
                 parameters={"script": None},
                 origin=True,
                 label="TAP Query",
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            label=label,
            inputs=inputs,
            outputs=outputs,
            origin=origin,
            description="A node that connects to a TAP service.",
        )
    
    def node_configure(self):
        if self.parameters['script'] is None:
            # write template script to node directory
            template_script = """# Example script for NodeTAPQuery

query["description"] = "Get 10 random objects"
query["adql"] = "SELECT TOP 10 objectId FROM dp1.Object"

"""         
            script_path = os.path.join(self.node_dir, f"script.py")

            os.makedirs(self.node_dir, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(template_script)
            self.parameters = {
                    "script": script_path
                }
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeTAPQuery"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir=d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        script = self.parameters.get("script", "")
        max_records = self.parameters.get("max_records", 3)

        query = {"adql": "", "description": ""}

        with open(script, "r") as f:
            code = f.read()
            exec(code, {"query": query, 
                        "parameters": self.parameters, 
                        })

        client = AstroosQueryLSST(root_dir=f"_pipelines/{self.node_id}", 
                      credentials_file=None,
                      max_records=max_records)

        print("Running TAP ADQL Query on LSST...")
        table = client.query_async(query["adql"])
        
        print(f"Number of results: {len(table)}")

        columns = {}
        for col in table.colnames:
            columns[col] = col
        # columns.pop("objectId", None)


        artifact = ArtifactItem(
                file_path=os.path.join(self.node_dir, "tap.fits"),
                dag=self.artifact_dag,
                node_id=self.node_id,
                )
        artifact.load_from_table(table, columns)
        artifact.materialize(self.node_id)

        self.outputs = [artifact]


class NodeLSSTButlerFetch(Node):
    def __init__(
            self,
            dag_dir=None,
            node_type="catalog_lsst_butler_fetch",
            node_id=None,
            parents=[],
            parameters={},
            label="Fetch LSST DP-1 Cutouts (Butler)",
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label=label,
            description="Use the RSP Butler service to fetch deep coadd cutouts.",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeLSSTButlerFetch"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        artifact = self.inputs[0]
        table = artifact.to_table(self.node_id) 

        print(f"Fetching LSST data via Butler for {len(table)} objects...")

        tasks = build_groups(
                table, 
                self.parameters.get("dataset")
                )

        with ProcessPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(worker_patch, tasks):
                pass

        self.outputs = [artifact]



def worker_patch(args):

    BANDS = ["u", "g", "r", "i", "z"]
    BANDS = ["u", "g", "r", "i", "z", "y"]

    # cutout stamp size (pixels)
    STAMP_W = 100
    STAMP_H = 100

    # tract, patch, object_rows, dataset_dir, dataset_labels = args
    tract, patch, object_rows, dataset_dict = args

    # print("object_rows")
    # print(object_rows)

    # dataset = FITS_Image_Morphometry_Photometry_Dataset(
            # dataset_dir=dataset_dir,
            # labels_init_file=dataset_labels,
            # N_bands=len(BANDS), 
            # N_morphometric_features=4,
            # N_photometric_features=4,
            # )
    dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(dataset_dict)

    from lsst.daf.butler import Butler
    butler = Butler("dp1", collections="LSSTComCam/DP1")

    # Load each band ONCE per patch
    coadds = {
        b: butler.get("deep_coadd", tract=tract, patch=patch, band=b)
        for b in BANDS
    }

    ext = geom.Extent2I(STAMP_W, STAMP_H)
    band_images = np.zeros((len(BANDS), STAMP_H, STAMP_W), dtype=np.float32)

    for row in tqdm(object_rows, desc=f"Processing cutouts", total=len(object_rows)): 

        
        # print("row\n\n")
        # print(row)
        ra_deg = float(row["ra"])
        dec_deg = float(row["dec"])

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
        # hdu_img.header['redshift'] = -999
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

def build_groups(objects, dataset_dict):
    groups = defaultdict(list)
    for row in objects:
        groups[(int(row["tract"]), int(row["patch"]))].append(row)
    return [(t, p, rows, dataset_dict) for (t, p), rows in groups.items()]




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


def make_cutout_header3(cutout, ra, dec):
    wcs = cutout.getWcs()

    hdr = fits.Header(wcs.getFitsMetadata().toDict())
    hdr["WCSAXES"] = 2

    width, height = cutout.getDimensions()

    x_ref = (width - 1) / 2.0
    y_ref = (height - 1) / 2.0

    sp = wcs.pixelToSky(geom.Point2D(x_ref, y_ref))
    ra_ref = sp.getRa().asDegrees()
    dec_ref = sp.getDec().asDegrees()

    hdr["CRPIX1"] = x_ref + 1.0
    hdr["CRPIX2"] = y_ref + 1.0
    hdr["CRVAL1"] = ra_ref
    hdr["CRVAL2"] = dec_ref

    ra_min, ra_max, dec_min, dec_max = wcs_bounds_radec(wcs, width, height)

    hdr["min_ra"] = ra_min
    hdr["max_ra"] = ra_max
    hdr["min_dec"] = dec_min
    hdr["max_dec"] = dec_max

    hdr["target_ra"] = float(ra)
    hdr["target_dec"] = float(dec)

    return hdr
