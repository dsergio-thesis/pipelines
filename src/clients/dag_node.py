
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    option_create = config.option_create
    node_type = config.node_type

    dag = PipelineDAG(label=pipeline_name)
    dag_dir = dag.dag_dir

    if option_create:

        if node_type == "bad_to_nan":
            dag_node = NodeBadToNaN(dag_dir=dag_dir)
        elif node_type == "script":
            dag_node = NodeScript(dag_dir=dag_dir)
        else: 
            dag_node = NodeGeneric(dag_dir=dag_dir)

        dag.add_node(dag_node)

    if input_artifact:
        dag.add_input_artifact(input_artifact)

    dag.to_yaml()


if __name__ == "__main__":
    main()
