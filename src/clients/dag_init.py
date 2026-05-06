
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name

    dag = PipelineDAG(label=pipeline_name)

if __name__ == "__main__":
    main()
