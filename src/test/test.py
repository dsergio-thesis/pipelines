import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from lsst.rsp.service import get_siav2_service
from lsst.rsp.utils import get_pyvo_auth
from pyvo.dal.adhoc import DatalinkResults


service = get_siav2_service("dp1")

target_ra = 52.74819403
target_dec = -27.8597582
search_radius = 0.2
circle = (target_ra, target_dec, search_radius)
results = service.search(
    pos=circle,
    calib_level=3,
)

for band in ['u', 'g', 'r', 'i', 'z']:
    table = results.to_table()
    tx = np.where((table['dataproduct_subtype'] == 'lsst.deep_coadd')
                & (table['lsst_band'] == band))[0]