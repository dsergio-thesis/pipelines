import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from lsst.rsp.service import get_siav2_service
from lsst.rsp.utils import get_pyvo_auth
from pyvo.dal.adhoc import DatalinkResults

import numpy as np
import matplotlib.pyplot as plt

import lsst.geom as geom

from lsst.rsp import get_tap_service
from lsst.rsp.utils import get_pyvo_auth
from lsst.rsp.service import get_siav2_service

import lsst.afw.display as afwDisplay
from lsst.afw.image import ExposureF
from lsst.afw.fits import MemFileManager
import lsst.afw.geom.ellipses as ellipses

from pyvo.dal.adhoc import DatalinkResults, SodaQuery

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.aperture import SkyEllipticalAperture

import galsim as gs
from lsst.gauss2d import Ellipse, EllipseMajor, Covariance


def pad_exposure_ml(exp, target=200, fill_value=0.0):
    """Pad or crop an ExposureF to a fixed size for ML, keeping the original WCS untouched."""
    h, w = exp.image.array.shape
    pad_x = max(target - w, 0)
    pad_y = max(target - h, 0)
    pad_left = pad_x // 2
    pad_top = pad_y // 2

    # New arrays
    new_img = np.full((target, target), fill_value, dtype=exp.image.array.dtype)
    new_mask = np.zeros((target, target), dtype=exp.mask.array.dtype)
    new_var = np.zeros((target, target), dtype=exp.variance.array.dtype)

    # Determine copy coordinates
    y0 = pad_top
    x0 = pad_left
    y1 = y0 + h
    x1 = x0 + w

    # Copy original data
    new_img[y0:y1, x0:x1] = exp.image.array
    new_mask[y0:y1, x0:x1] = exp.mask.array
    new_var[y0:y1, x0:x1] = exp.variance.array

    # Create a new ExposureF without WCS modification
    new_exp = ExposureF(target, target)
    new_exp.image.array[:, :] = new_img
    new_exp.mask.array[:, :] = new_mask
    new_exp.variance.array[:, :] = new_var

    # Keep original WCS for plotting
    new_exp.setWcs(exp.getWcs())
    new_exp.setPhotoCalib(exp.getPhotoCalib())

    return new_exp

def get_cutout_bands(target_ra, target_dec, bands):


    service = get_siav2_service("dp1")

    
    spherePoint = geom.SpherePoint(target_ra*geom.degrees, target_dec*geom.degrees)

    search_radius = 0.005
    circle = (target_ra, target_dec, search_radius)
    results = service.search(
        pos=circle,
        calib_level=3,
    )

    band_images = {}

    for band in bands: 
        table = results.to_table()
        tx = np.where((table['dataproduct_subtype'] == 'lsst.deep_coadd')
                    & (table['lsst_band'] == band))[0]

        print("tx: ", tx)

        datalink_url = results[tx[0].item()].access_url
        dl = DatalinkResults.from_result_url(datalink_url, session=get_pyvo_auth())

        sq = SodaQuery.from_resource(dl,
                                     dl.get_adhocservice_by_id("cutout-sync-exposure"),
                                     session=get_pyvo_auth())

        sq.circle = (spherePoint.getRa().asDegrees() * u.deg,
                     spherePoint.getDec().asDegrees() * u.deg,
                     search_radius * u.deg)
        cutout_bytes = sq.execute_stream().read()
        mem = MemFileManager(len(cutout_bytes))
        mem.setData(cutout_bytes, len(cutout_bytes))

        cutout = ExposureF(mem)
        cutout = pad_exposure_ml(cutout)
        band_images[band] = cutout
        print(f"cutout image shape: {cutout.getImage().getDimensions()}")
        print(f"cutout type: {type(cutout)}")
        print(f"cutout metadata: {cutout.getMetadata()}")
        print(f"Retrieved {band}-band cutout image.")

    return band_images
