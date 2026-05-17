
from clients._local import *

def main():

    config = client_config()

    dag = PipelineDAG()

    print(dag.dag_id)

if __name__ == "__main__":
    main()

