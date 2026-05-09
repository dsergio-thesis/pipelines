
from clients._local import *

def main():

    config = client_config()

    dag = PipelineDAG()

    dag.status()

if __name__ == "__main__":
    main()

