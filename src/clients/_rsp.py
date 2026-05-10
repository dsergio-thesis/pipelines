
import sys
import os
import importlib

from astroos_pipelines.dag import *
from astroos_pipelines.lsst.dag import NodeTAPQuery 
from astroos_pipelines.hst.dag import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.logger.logger import setup_logging
import logging


def client_config():
    importlib.reload(sys.modules['astroos_pipelines.dag'])
    importlib.reload(sys.modules['astroos_pipelines.lsst.dag'])
    importlib.reload(sys.modules['astroos_pipelines.hst.dag'])
    importlib.reload(sys.modules['astroos_pipelines.datasets'])
    importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
    importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
    setup_logging()
    log = logging.getLogger(__name__)

    config = AstroosConfig.from_cli()

    return config
