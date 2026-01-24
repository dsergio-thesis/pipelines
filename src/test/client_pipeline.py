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

if (sys.modules.get('src.astroos.query') is not None):
    del sys.modules['src.astroos.query']
if (sys.modules.get('src.astroos.fetch') is not None):
    del sys.modules['src.astroos.fetch']
if (sys.modules.get('src.astroos.pipelines') is not None):
    del sys.modules['src.astroos.pipelines']
if (sys.modules.get('src.astroos.datasets') is not None):
    del sys.modules['src.astroos.datasets']
if (sys.modules.get('src.astroos.transforms') is not None):
    del sys.modules['src.astroos.transforms']
if (sys.modules.get('src.astroos.utils') is not None):
    del sys.modules['src.astroos.utils']

from src.astroos.fetch import AstroosFetchSDSS
from src.astroos.query import AstroosQueryNED, \
    AstroosQuerySimbad, AstroosQuerySDSS, \
    AstroQueryUtils as aq_utils
from src.astroos.pipelines import StageCatalogSDSS, \
    StageFilterCatalogSDSS, \
    StageCatalogSDSS_V2, \
    StageFetchSDSS_V2_ManualCutout, StageFetchSDSS_V2_AutoCutout, \
    StageFetchSDSS, StageFetchSDSS_V3_ManualCutout, \
    PipelineClassification, StageCatalogLSST,  StageFetchLSSTSoda, PipelineDummy
from src.astroos.datasets import FITS_Image_Features_Dataset

from src.astroos.utils.utils import plot_random_samples_from_dataset

from src.astroos.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid



if __name__ == "__main__":

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
    
    print(f"Max records to fetch: {max_records}")

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
        PipelineDummy(
            name="dummy_pipeline",
        ),
        PipelineClassification(
            name=name,
            metadata=pipeline_metadata,
            max_records=max_records,
            dataset=dataset_cartesian,
            minor_version="a3",
        ),
    ]

    pipelines[0].add_stages([
    ])

    # pipelines[1].add_stages([
    #     StageCatalogSDSS_V2(),
    #     StageFetchSDSS_V3_ManualCutout(dataset_cartesian),
    # ])

    pipelines[1].add_stages([
        StageCatalogLSST(),
        StageFetchLSSTSoda(dataset_cartesian),
    ])

    for p in pipelines:
        p.run_pipeline()

