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

service = get_siav2_service("dp1")


target_ra = 52.74819403
target_dec = -27.8597582

spherePoint = geom.SpherePoint(target_ra*geom.degrees, target_dec*geom.degrees)

search_radius = 0.01
circle = (target_ra, target_dec, search_radius)
results = service.search(
    pos=circle,
    calib_level=3,
)

bands = ['u', 'g', 'r', 'i', 'z']
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
    band_images[band] = cutout
    print(f"Retrieved {band}-band cutout image.")

# Display the images using matplotlib
fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
for ax, band in zip(axes, bands):
    image = band_images[band].getMaskedImage().getImage()
    ax.set_title(f'{band}-band')
    ax.axis('off')

    ax.imshow(image.array, origin='lower', cmap='gray')
    # save each image
    plt.imsave(f'cutout_{band}_band.png', image.array, cmap='gray')
