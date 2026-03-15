
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    pipelines = [
            PipelineClassification(
                name=pipeline_name,
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=None,
                minor_version=None,
                ),
            ]
            
    pipelines[0].add_stages([ # EDA
        StageCatalogLSST(),
        StageLSSTExploratoryDataAnalysis(),
        ])

    pipelines[0].run_pipeline()

if __name__ == "__main__":
    main()
