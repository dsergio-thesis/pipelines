
from clients._local import *

from astroos_pipelines.hst.dag import HSTNodeCatalog

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    parameter = config.parameter
    option_create = config.option_create
    node_type = config.node_type
    max_records = config.max_records
    
    print("registry keys", Node.registry.keys())
    dag = PipelineDAG(label=pipeline_name)
    dag_dir = dag.dag_dir

    if option_create:

        if node_type == "bad_to_nan":
            dag_node = NodeBadToNaN(dag_dir=dag_dir)
        elif node_type == "script":
            dag_node = NodeScript(dag_dir=dag_dir)
        elif node_type == "input":
            dag_node = NodeInput(dag_dir=dag_dir)
        elif node_type == "export":
            dag_node = NodeExport(dag_dir=dag_dir)
        elif node_type == "hst-catalog":
            dag_node  = HSTNodeCatalog(
                parameters={"max_records": max_records}, 
                dag_dir=dag_dir) 

        else: 
            dag_node = NodeGeneric(dag_dir=dag_dir)

        dag.add_node(dag_node)

    if input_artifact:
        dag.add_input_artifact_item(input_artifact)
    if parameter:
        dag.add_parameter(parameter)

    dag.to_yaml()


if __name__ == "__main__":
    main()
