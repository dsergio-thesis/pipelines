
from clients._rsp import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name

    dag = PipelineDAG(label=pipeline_name, new=True)
    dag.to_yaml()
    dag.status()

if __name__ == "__main__":
    main()

