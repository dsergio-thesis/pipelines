
import sys
import os
import importlib

from astroos_pipelines.pipelines import PipelineClassification
from astroos_pipelines.hst.pipelines import StageHSTCatalogQuery, \
        StageHSTExploratoryDataAnalysis
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.logger.logger import setup_logging
import logging


def client_config():
    # global configuration for development and testing
    importlib.reload(sys.modules['astroos_pipelines.pipelines'])
    importlib.reload(sys.modules['astroos_pipelines.datasets'])
    importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
    importlib.reload(sys.modules['astroos_pipelines.logger.logger'])

    setup_logging()
    log = logging.getLogger(__name__)
    
    config = AstroosConfig.from_cli()
    coord, radius = config.get_target("CDF_South");

    print()
    print(config)

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}
    print(pipeline_metadata)

    return config, pipeline_metadata

