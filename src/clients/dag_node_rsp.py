
from clients._rsp import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    parameter = config.parameter
    option_create = config.option_create
    node_type = config.node_type
    node_label = config.node_label
    max_records = config.max_records

    print(f"Label: {node_label}")
    
    dag = PipelineDAG(label=pipeline_name)
    if not dag.is_initialized():
        print("No pipelines found.")
        PipelineDAG.usage()
        return

    dag_dir = dag.dag_dir

    if option_create:
        if node_type == "tap":
            dag_node = NodeTAPQuery(label=node_label)
            dag.add_node(dag_node)

    if parameter:
        dag.add_parameter(parameter)

    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
