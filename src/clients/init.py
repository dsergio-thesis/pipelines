
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    max_records = config.max_records
    pipeline_name = config.pipeline_name
    print("Configuration loaded successfully. Pipeline name:", pipeline_name)

    # root = NodeRoot()
    # label = "LSST/HST DAG"
    # dag = PipelineDAG(label=label)
    # dag.add_node(root)
    # dag.to_graphviz()
    # dag.to_yaml("_pipelines/lsst_hst_img.yaml")

if __name__ == "__main__":
    main()
