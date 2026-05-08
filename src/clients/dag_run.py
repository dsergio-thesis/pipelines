
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name

    if pipeline_name is not None:
        print(f"Running pipeline: {pipeline_name}")
        dag = PipelineDAG(pipeline_name=pipeline_name)
    else:
        print("No pipeline name provided. Running active pipeline.")
        dag = PipelineDAG()
    dag.run()
    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
