
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name

    dag = PipelineDAG()
    dag.run()
    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
