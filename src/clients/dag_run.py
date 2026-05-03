
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    pipeline_name = config.pipeline_name
    print("Configuration loaded successfully.")
    print("Pipeline name:", pipeline_name)

    dag = PipelineDAG()

    dag.print_head()

    dag.run()

    dag.to_yaml()

    dag.to_graphviz()
    # dag.to_yaml("_pipelines/lsst_hst_img.yaml")

if __name__ == "__main__":
    main()
