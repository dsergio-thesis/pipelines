
import sys
import os
import importlib

from astroos_pipelines.pipelines import PipelineClassification, PipelineDummy
from astroos_pipelines.lsst.pipelines import StageCatalogLSST,  StageFetchLSSTSoda, \
        StageMatchLSSTtoHST, StagePreprocessLSST
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig

importlib.reload(sys.modules['astroos_pipelines.lsst.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
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

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}
    print(pipeline_metadata)


    dataset_cart_cutouts_morph = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            N_bands=5, 
            N_morphometric_features=4,
            N_photometric_features=4,
            )

    dataset_cart_phot = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name + "_phot"),
            labels_init_file=label_def_file,
            N_bands=5, 
            N_morphometric_features=0,
            N_photometric_features=4,
            )

    pipelines = [
            PipelineDummy(
                name="dummy_pipeline",
                ),
            PipelineClassification(
                name=pipeline_name,
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset_cart_cutouts_morph,
                minor_version=pipeline_minor_version,
                ),
            PipelineClassification(
                name=pipeline_name + "_phot",
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset_cart_phot,
                minor_version=pipeline_minor_version,
                ),
            ]

            
    pipelines[0].add_stages([])

    pipelines[1].add_stages([
        StageCatalogLSST(),
        StageMatchLSSTtoHST(),
        StagePreprocessLSST(),
        StageFetchLSSTSoda(),
        ])

    pipelines[2].add_stages([
        StageCatalogLSST(),
        StageMatchLSSTtoHST(),
        StagePreprocessLSST(),
        ])

    # pipelines[1].run_pipeline()
    pipelines[2].run_pipeline()

    # for p in pipelines:
    #     p.run_pipeline()

if __name__ == "__main__":
    main()
