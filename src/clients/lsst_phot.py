
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")


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
                minor_version=None,
                ),
            PipelineClassification(
                name=pipeline_name + "_phot",
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset_cart_phot,
                minor_version=None,
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
