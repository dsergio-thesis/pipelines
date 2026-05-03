
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    print("Configuration loaded successfully.")

    PipelineDAG.list_dags()


if __name__ == "__main__":
    main()
