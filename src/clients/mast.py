from torchvision import transforms
import sys
import os
import importlib

# astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u

from astroos_pipelines.sdss.pipelines import StageCatalogSDSS_V2, StageFetchSDSS_V3_ManualCutout
from astroos_pipelines.pipelines import PipelineClassification, PipelineDummy
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from config.astroos_config import AstroosConfig
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid
from astroos_pipelines.mast.query import AstroosQueryMast

importlib.reload(sys.modules['astroos_pipelines.sdss.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['config.astroos_config'])
importlib.reload(sys.modules['astroos_pipelines.mast.query'])

import sys
import importlib
from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)

def main():

    config = AstroosConfig.from_cli()
    coord, radius = config.get_target("CDF_South")
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version
    max_records = config.max_records
    label_def_file = config.label_def_file
    print()
    print(config)

    
    params = {
        "ra": coord.ra.deg,
        "dec": coord.dec.deg,
        "radius": radius
    }
    
    mast = AstroosQueryMast(root_dir = "test_data", res_object_identifier_column="obsid") 
    res = mast.query(query_params=params)
    print(res)
    #print column names
    print(res.columns)



if __name__ == "__main__":
    main()
