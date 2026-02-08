from torchvision import transforms
import sys
import os
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
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.utils.utils import plot_random_samples_from_dataset
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
importlib.reload(sys.modules['astroos_pipelines.utils.utils'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['config.pipeline_config'])

def main():

    config = PipelineConfig.from_env()
    coord, radius = config.get_target("CDF_South")

    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version

    print(f"Config summary: ")
    print(f" - Dataset directory: {dataset_dir}")
    print(f" - Dataset name: {dataset_name}")
    print(f" - Pipeline name: {pipeline_name}")
    print(f" - Pipeline minor version: {pipeline_minor_version}")
    print(f" - Coordinates: {coord}, and radius: {radius}")

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

    if len(sys.argv) > 1:
        max_records = int(sys.argv[1])
    else:
        max_records = 3

    print(f"Max records to fetch: {max_records}")

    pipeline_metadata = {
            'query_coords': coord,
            'query_radius': radius,
            }

    labels_file = "./sdss_morph_types_info.csv"

    dataset_cart_lsst = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=labels_file,
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
                minor_version=None,
                )
            ]

    pipelines[0].add_stages([
        ])

    pipelines[1].add_stages([
        StageCatalogLSST(),
        StageFetchLSSTSoda(dataset_cart_lsst),
        ])

    pipelines[1].run_pipeline()

    # for p in pipelines:
    #     p.run_pipeline()

if __name__ == "__main__":
    main()
