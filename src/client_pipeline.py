import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from skimage import morphology
from PIL import Image
import pandas as pd

# astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u

if (sys.modules.get('query') is not None):
    del sys.modules['query']
if (sys.modules.get('fetch') is not None):
    del sys.modules['fetch']
if (sys.modules.get('pipelines') is not None):
    del sys.modules['pipelines']
if (sys.modules.get('datasets') is not None):
    del sys.modules['datasets']
if (sys.modules.get('transforms') is not None):
    del sys.modules['transforms']
if (sys.modules.get('utils') is not None):
    del sys.modules['utils']

from fetch import AstroosFetchSDSS
from query import AstroosQueryNED, \
    AstroosQuerySimbad, AstroosQuerySDSS, \
    AstroQueryUtils as aq_utils
from pipelines import StageCatalogSDSS, \
    StageFilterCatalogSDSS, \
    StageCatalogSDSS_V2, \
    StageFetchSDSS_V2_ManualCutout, StageFetchSDSS_V2_AutoCutout, \
    StageFetchSDSS, StageFetchSDSS_V3_ManualCutout, \
    PipelineClassification
from datasets import FITS_Image_Features_Dataset

from utils import plot_random_samples_from_dataset

from transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

transformPolar = transforms.Compose([
    # transforms.ToTensor(),
    # AddGaussianNoise(mean=0., std=0.3),
    transforms.CenterCrop(80),
    CropAroundCentroid(crop_size=(100, 100)),
    SegmentationTransform(nsigma=0.5, min_area=40),
    PolarTransform(output_size=(20, 20)),
    # CropZeros(),
])
transformCartesian = transforms.Compose([
    # transforms.ToTensor(),
    # AddGaussianNoise(mean=0., std=0.3),
    # transforms.CenterCrop(80),
    # CropAroundCentroid(crop_size=(100, 100)),
    # SegmentationTransform(nsigma=0.2, min_area=40),
    # CropAroundCentroid(crop_size=(30, 30)),
    # CropAroundCentroid(crop_size=(20, 20)),
    # SegmentationTransform(nsigma=0.2, min_area=40),
])

# Example coordinates:

# Format HH MM SS.SSS +DD MM SS.SS
virgo_cluster =  '12 30 49.423 +12 23 28.04'
obj_3c273 = '12 29 06.699 +02 03 08.60'
someother = '14 35 42.8685615528 +40 18 02.133470196'

query_obj = someother

# get from args if provided
if len(sys.argv) > 1:
    max_records = int(sys.argv[1])
else:
    max_records = 3

coords = SkyCoord(query_obj, frame=ICRS, unit=(u.hourangle, u.deg))
radius_hour = u.hourangle *     0
radius_min =  u.arcmin *        40
radius_sec =  u.arcsec *        20
radius = radius_hour + radius_min + radius_sec

pipeline_metadata = {
    'query_coords': coords,
    'query_radius': radius,
}

name = "p5"
labels_file = "./sdss_morph_types_info.csv"

dataset_cartesian = FITS_Image_Features_Dataset(
    dir="./data/demo2",
    labels_init_file="./sdss_morph_types_info.csv",
    N_bands=5, 
    N_features=4, 
    transform=transformCartesian,
    photometric_transform=MorphometryFeatures()
)

pipelines = [
    PipelineClassification(
        name=name,
        metadata=pipeline_metadata,
        max_records=max_records,
        dataset=dataset_cartesian,
        minor_version=None,
    ),
]

pipelines[0].add_stages([
    StageCatalogSDSS_V2(),
    StageFetchSDSS_V3_ManualCutout(dataset_cartesian),
])

for p in pipelines:
    p.run_pipeline()

