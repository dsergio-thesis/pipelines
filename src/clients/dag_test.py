
from clients._local import *

def main():

    config = client_config()
    dag_dir = config.pipeline_dir


    dag = PipelineDAG(new=True)
    n1 = NodeGeneric()
    n2 = NodeGeneric(parents=[n1.node_id])
    n3 = NodeGeneric(parents=[n2.node_id])

    dag.add_node(n1)
    dag.add_node(n2)
    dag.add_node(n3)
    
    dag.run()
    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
