
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    parameter = config.parameter
    option_create = config.option_create
    option_origin = config.option_origin
    parent = config.parent
    node_type = config.node_type
    node_label = config.node_label
    max_records = config.max_records

    print(f"Label: {node_label}")
    
    dag = PipelineDAG(label=pipeline_name)
    if not dag.is_initialized():
        print("No pipelines found.")
        PipelineDAG.usage()
        return

    parent_id = dag.get_node_id(parent) if parent else None

    dag_dir = dag.dag_dir

    if option_create:
        if node_type == "script":
            dag_node = NodeScript(label=node_label)
        elif node_type == "import":
            dag_node = NodeImport(label=node_label, parameters={"max_records": max_records})
        elif node_type == "export":
            dag_node = NodeExport(label=node_label)
        elif node_type == "eda":
            dag_node = NodeEDA(label=node_label)
        elif node_type == "eda-script":
            dag_node = NodeEDAScript(label=node_label)
        else: 
            dag_node = NodeGeneric(label=node_label)
            if parent_id:
                dag_node.parents = [parent_id]

        dag.add_node(dag_node)
        if option_origin:
            dag_node.parents = []
    
    if parent_id:
        dag.head.parents.append(parent_id)

    if input_artifact:
        dag.add_input_artifact_item(input_artifact)
    if parameter:
        dag.add_parameter(parameter)

    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
