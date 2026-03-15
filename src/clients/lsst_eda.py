
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")
    
    catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": pipeline_metadata.get("query_coords", None),
                "query_radius": pipeline_metadata.get("query_radius", None),})

    eda = LSSTNodeEDA(
            parameters={"max_records": max_records,},
            parents=[catalog.node_id],
            )


    dag = PipelineDAG()

    dag.add_node(catalog)
    dag.add_node(eda)
    dag.run_from_node(catalog.node_id)
    dag.to_yaml("_pipelines/lsst_eda_pipeline.yaml")


    # pipelines = [
            # PipelineClassification(
                # name=pipeline_name,
                # metadata=pipeline_metadata,
                # max_records=max_records,
                # dataset=None,
                # minor_version=None,
                # ),
            # ]

    # pipelines[0].add_stages([ # EDA
                             # StageCatalogLSST(),
                             # StageLSSTExploratoryDataAnalysis(),
                             # ])
                             
    # pipelines[0].run_pipeline()

if __name__ == "__main__":
    main()
