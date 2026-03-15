
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    dataset = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            N_bands=5, 
            N_morphometric_features=4,
            N_photometric_features=4,
            )

    pipelines = [
            PipelineClassification(
                name=pipeline_name,
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset,
                minor_version=None,
                ),
            ]

    pipelines[0].add_stages([ # EDA
                             StageHSTCatalogQuery(),
                             StageHSTExploratoryDataAnalysis(),
                             ])

    pipelines[0].run_pipeline()

if __name__ == "__main__":
    main()
