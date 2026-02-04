import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from skimage import morphology
from PIL import Image
import pandas as pd
import importlib

# astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u


from astroos_pipelines.query import AstroosQueryNED, \
    AstroosQuerySimbad, AstroosQuerySDSS, \
    AstroQueryUtils as aq_utils
from astroos_pipelines.fetch import AstroosFetchSDSS
from astroos_pipelines.pipelines import StageCatalogSDSS, \
    StageFilterCatalogSDSS, \
    StageCatalogSDSS_V2, \
    StageFetchSDSS_V2_ManualCutout, StageFetchSDSS_V2_AutoCutout, \
    StageFetchSDSS, StageFetchSDSS_V3_ManualCutout, \
    PipelineClassification, StageCatalogLSST,  StageFetchLSSTSoda, PipelineDummy
from astroos_pipelines.datasets import FITS_Image_Features_Dataset
from astroos_pipelines.utils.utils import plot_random_samples_from_dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules['astroos_pipelines.fetch'])
importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.utils.utils'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])


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

    name = "p7"
    labels_file = "./sdss_morph_types_info.csv"

    dataset_cart_lsst = FITS_Image_Features_Dataset(
        dir="./data/lsst-4",
        labels_init_file="./sdss_morph_types_info.csv",
        N_bands=5, 
        N_features=0, 
        transform=transformCartesian,
        photometric_transform=None
    )

    dataset_cart_sdss = FITS_Image_Features_Dataset(
        dir="./data/sdss-1",
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
            dataset=dataset_cart_lsst,
            minor_version=None,
        ),
        PipelineClassification(
            name=name,
            metadata=pipeline_metadata,
            max_records=max_records,
            dataset=dataset_cart_sdss,
            minor_version="sdss-1",
        ),
    ]

    pipelines[0].add_stages([
    ])

    pipelines[1].add_stages([
        StageCatalogLSST(),
        StageFetchLSSTSoda(dataset_cart_lsst),
    ])

    pipelines[2].add_stages([
        StageCatalogSDSS_V2(),
        StageFetchSDSS_V3_ManualCutout(dataset_cart_sdss),
    ])

    pipelines[1].run_pipeline()

    # for p in pipelines:
    #     p.run_pipeline()

