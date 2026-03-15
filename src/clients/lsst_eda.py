
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")
    
    lsst_catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": pipeline_metadata.get("query_coords", None),
                "query_radius": pipeline_metadata.get("query_radius", None),})
    lsst_eda = LSSTNodeEDA(
            parameters={"max_records": max_records,},
            parents=[lsst_catalog.node_id],
            )
    lsst_preprocess = LSSTNodePreprocess(
            parameters="max_records": max_records,
            parents=[lsst_eda.node_id],
            )
    lsst_preprocess_eda = LSSTNodeEDA(
            parameters={"max_records": max_records,},
            parents=[lsst_preprocess.node_id],
            )


    dag = PipelineDAG()

    dag.add_node(lsst_catalog)
    dag.add_node(lsst_eda)
    dag.add_node(lsst_preprocess)
    dag.add_node(lsst_preprocess_eda)

    dag.run_from_node(lsst_catalog.node_id)
    dag.to_yaml("_pipelines/lsst_eda_pipeline.yaml")


if __name__ == "__main__":
    main()
