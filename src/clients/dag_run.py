
from clients._local import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    start_node_id = config.start_node_id
    

    dag = PipelineDAG(label=pipeline_name)
    if not dag.is_initialized():
        print("No pipelines found.")
        PipelineDAG.usage()
        return

    if start_node_id:
        print(f"Starting pipeline '{pipeline_name}' from node '{start_node_id}'...")
        dag.run(start_node_id=start_node_id, check_dependencies=False)
    else:
        print(f"Starting pipeline '{pipeline_name}' from the head...")
        dag.run()
    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
