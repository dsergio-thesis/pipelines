from networkx import radius
from torchvision import transforms
import sys
import os
import importlib

# astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u

from astroos_pipelines.query import AstroosQuerySDSS
from astroos_pipelines.fetch import AstroosFetchSDSS
from astroos_pipelines.pipelines import StageCatalogSDSS, \
        StageFilterCatalogSDSS, \
        StageCatalogSDSS_V2, \
        StageFetchSDSS_V2_ManualCutout, StageFetchSDSS_V2_AutoCutout, \
        StageFetchSDSS, StageFetchSDSS_V3_ManualCutout, \
        PipelineClassification, StageCatalogLSST,  StageFetchLSSTSoda, PipelineDummy
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
        MorphometryFeatures, \
        SegmentationTransform, \
        PolarTransform, \
        CropZeros, \
        CropAroundCentroid
from config.pipeline_config import PipelineConfig

importlib.reload(sys.modules['astroos_pipelines.fetch'])
importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['config.pipeline_config'])

from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)

def main():

    config = PipelineConfig.from_cli()
    coord, radius = config.get_target("CDF_South")
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version
    max_records = config.max_records
    label_def_file = config.label_def_file

    print()
    print(config)

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}

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

    dataset_cart_lsst = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            N_bands=5, 
            N_morphometric_features=0,
            N_photometric_features=4,
            transform=transformCartesian,
            morphometric_transform=MorphometryFeatures(),
            photometric_transform=None
            )

    pipelines = [
            PipelineDummy(
                name="dummy_pipeline",
                ),
            PipelineClassification(
                name=pipeline_name,
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset_cart_lsst,
                minor_version=pipeline_minor_version,
                )
            ]

    pipelines[0].add_stages([])

    pipelines[1].add_stages([
        StageCatalogLSST(),
        StageFetchLSSTSoda(dataset_cart_lsst),
        ])

    pipelines[0].run_pipeline()

    # for p in pipelines:
    #     p.run_pipeline()

if __name__ == "__main__":
    main()
