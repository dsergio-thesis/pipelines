
# LSST RSP mode
# if running in RSP mode, import LSST RSP dependencies
import sys


def init_rsp_mode():

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
    except ImportError:
        print("LSST RSP mode disabled.")

    return rsp_mode
