
import sys
import os
import importlib

from astroos_pipelines.pipelines import PipelineClassification
from astroos_pipelines.lsst.pipelines import * 
from astroos_pipelines.lsst.dag import *
from astroos_pipelines.dag import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.logger.logger import setup_logging
import logging


def client_config():
    # global config
    importlib.reload(sys.modules['astroos_pipelines.lsst.pipelines'])
    importlib.reload(sys.modules['astroos_pipelines.lsst.dag'])
    importlib.reload(sys.modules['astroos_pipelines.pipelines'])
    importlib.reload(sys.modules['astroos_pipelines.datasets'])
    importlib.reload(sys.modules['astroos_pipelines.dag'])
    importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
    importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
    setup_logging()
    log = logging.getLogger(__name__)

    config = AstroosConfig.from_cli()
    print("Config: ")
    print(config)
    coord, radius = config.get_target("Extended Chandra Deep Field South (ECDFS)")
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version
    max_records = config.max_records
    label_def_file = config.label_def_file

    print()
    print(config)

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}
    print(pipeline_metadata)

    return config, pipeline_metadata
