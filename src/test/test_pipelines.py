import sys
import importlib
from astroos_pipelines.pipelines import PipelineClassification
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])

def test_pipeline_classification():
    dataset = FITS_Image_Morphometry_Photometry_Dataset(dataset_dir="test_data")
    pipeline = PipelineClassification(name="test", max_records=10, dataset=dataset)
    assert pipeline is not None

def test_dataset_loading():
    dataset = FITS_Image_Morphometry_Photometry_Dataset(dataset_dir="test_data")
    assert len(dataset) is  0
