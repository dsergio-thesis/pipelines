import os
import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from astropy.io import fits


from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.utils.plots.as_image import plot_random_samples_as_image
from astroos_pipelines.utils.plots.as_html import plot_random_samples_as_html
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_image'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_html'])

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



def main():
    pass



if __name__ == "__main__":
    main()
