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

from test3 import get_cutout_bands


service = get_siav2_service("dp1")


target_ra = 52.74819403
target_dec = -27.8597582



bands = ['u', 'g']
band_images = get_cutout_bands(target_ra, target_dec, bands)

# Display the images using matplotlib
fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
for ax, band in zip(axes, bands):
    image = band_images[band].getMaskedImage().getImage()
    print(f"type of image.array: {type(image.array)}, shape: {image.array.shape}")
    ax.set_title(f'{band}-band')
    ax.axis('off')

    ax.imshow(image.array, origin='lower', cmap='gray')
    # save each image
    plt.imsave(f'cutout_{band}_band.png', image.array, cmap='gray')
