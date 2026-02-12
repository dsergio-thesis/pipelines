import sys
import importlib
from astroos_pipelines.pipelines import PipelineClassification
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from config.astroos_config import AstroosConfig

importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['config.astroos_config'])

def test_pipeline_classification_not_None():
    dataset = FITS_Image_Morphometry_Photometry_Dataset(dataset_dir="test_data")
    pipeline = PipelineClassification(name="test", max_records=10, dataset=dataset)
    assert pipeline is not None

