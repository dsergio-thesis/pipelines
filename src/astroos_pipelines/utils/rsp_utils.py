
# LSST RSP mode
# if running in RSP mode, import LSST RSP dependencies
import sys


def init_rsp():

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
    except ModuleNotFoundError:
        print("LSST RSP mode disabled.")

    return rsp_mode


import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from lsst.rsp.service import get_siav2_service
from lsst.rsp.utils import get_pyvo_auth
from pyvo.dal.adhoc import DatalinkResults

import numpy as np
import matplotlib.pyplot as plt

from pyvo.dal.adhoc import DatalinkResults, SodaQuery

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.aperture import SkyEllipticalAperture

import galsim as gs

try:
    from lsst.gauss2d import Ellipse, EllipseMajor, Covariance
    import lsst.geom as geom
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    import lsst.afw.display as afwDisplay
    from lsst.afw.image import ExposureF
    from lsst.afw.fits import MemFileManager
    import lsst.afw.geom.ellipses as ellipses
except:
    print("RSP not supported")


def pad_exposure_ml(exp, target=200, fill_value=0.0):
    """Pad or crop an ExposureF to a fixed size, re-centering WCS on RA/Dec."""

    img = exp.image.array
    h, w = img.shape

    # Compute padding offsets
    pad_x = max(target - w, 0)
    pad_y = max(target - h, 0)
    x0 = pad_x // 2
    y0 = pad_y // 2

    # Allocate new arrays
    new_img = np.full((target, target), fill_value, dtype=img.dtype)
    new_mask = np.zeros((target, target), dtype=exp.mask.array.dtype)
    new_var = np.zeros((target, target), dtype=exp.variance.array.dtype)

    # Copy data
    new_img[y0:y0+h, x0:x0+w] = img
    new_mask[y0:y0+h, x0:x0+w] = exp.mask.array
    new_var[y0:y0+h, x0:x0+w] = exp.variance.array

    # Create new exposure
    new_exp = ExposureF(target, target)
    new_exp.image.array[:, :] = new_img
    new_exp.mask.array[:, :] = new_mask
    new_exp.variance.array[:, :] = new_var

    # --- WCS FIX ---
    wcs = exp.getWcs()
    if wcs is not None:
        wcs = wcs.clone()

        # Shift reference pixel by padding offset
        shift = geom.Extent2D(x0, y0)
        wcs = wcs.shiftReferencePixel(shift)

        new_exp.setWcs(wcs)

    new_exp.setPhotoCalib(exp.getPhotoCalib())

    return new_exp

def get_cutout_bands(target_ra, target_dec, bands = ['u','g','r','i','z','y']):


    service = get_siav2_service("dp1")

    
    spherePoint = geom.SpherePoint(target_ra*geom.degrees, target_dec*geom.degrees)

    search_radius = 0.005
    circle = (target_ra, target_dec, search_radius)
    results = service.search(
        pos=circle,
        calib_level=3,
    )

    band_images = np.empty((len(bands), 200, 200)) 
    r = np.random.rand()

    for i, band in enumerate(bands): 
        table = results.to_table()
        tx = np.where((table['dataproduct_subtype'] == 'lsst.deep_coadd')
                    & (table['lsst_band'] == band))[0]

        # print("tx: ", tx)

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
        image = cutout.getImage().getArray()

        band_images[i] = image

        # save as grayscale image
        # plt.imsave(f"images/cutout_{r}_{band}.png", image, cmap='gray')
        # print(f"cutout image shape: {cutout.getImage().getDimensions()}")
        # print(f"cutout type: {type(cutout)}")
        # print(f"cutout metadata: {cutout.getMetadata()}")
        # print(f"Retrieved {band}-band cutout image. nans: {np.sum(np.isnan(image))}, inf: {np.sum(np.isinf(image))}")

    return band_images
