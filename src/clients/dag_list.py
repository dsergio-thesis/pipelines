
from clients._local import *

def main():

    config = client_config()

    PipelineDAG.list_dags()

if __name__ == "__main__":
    main()
