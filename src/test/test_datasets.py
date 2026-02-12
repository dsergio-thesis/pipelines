
import importlib
import sys
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
importlib.reload(sys.modules['astroos_pipelines.datasets'])

def test_dataset_not_None():
    dataset = FITS_Image_Morphometry_Photometry_Dataset(dataset_dir="test_data")
    assert dataset is not None 

