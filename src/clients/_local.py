
import sys
import os
import importlib
import cmcrameri.cm as cmc
from torchvision import transforms

from astroos_pipelines.utils.plots.as_image import plot_random_samples_as_image
from astroos_pipelines.utils.plots.as_html import plot_random_samples_as_html
from astroos_pipelines.pipelines import PipelineClassification, \
        StageCatalogRandom, StageTransformRandom
from astroos_pipelines.hst.pipelines import StageHSTCatalogQuery, \
        StageHSTExploratoryDataAnalysis
from astroos_pipelines.dag import PipelineDAG, Node, Artifact
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.logger.logger import setup_logging
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid
import logging


def client_config():
    # global configuration for development and testing
    importlib.reload(sys.modules['astroos_pipelines.pipelines'])
    importlib.reload(sys.modules['astroos_pipelines.dag'])
    importlib.reload(sys.modules['astroos_pipelines.datasets'])
    importlib.reload(sys.modules['astroos_pipelines.transforms'])
    importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
    importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
    importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_image'])
    importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_html'])

    setup_logging()
    log = logging.getLogger(__name__)
    
    config = AstroosConfig.from_cli()
    coord, radius = config.get_target("Extended Chandra Deep Field South (ECDFS)");

    print()
    print(config)

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}
    print(pipeline_metadata)

    return config, pipeline_metadata

