
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    pipeline_name = config.pipeline_name
    print("Configuration loaded successfully. Pipeline name:", pipeline_name)

    dag = PipelineDAG(label=pipeline_name)
    # root = NodeRoot()
    # dag.add_node(root)
    # dag.to_graphviz()
    # dag.to_yaml("_pipelines/lsst_hst_img.yaml")

if __name__ == "__main__":
    main()
