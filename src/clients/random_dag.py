
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    node1 = NodeCatalogRandom(
            node_type='catalog_random', 
            parameters={'max_records': max_records},
            )

    node2 = NodeTransformRandom(
            node_type='transform_random', 
            parents=[node1.node_id],
            parameters={'max_records': max_records},
            )
    dag = PipelineDAG()
    dag.add_node(node1)
    dag.add_node(node2)
            
    print(f"Running DAG with node: {node1.node_id}")
    dag.run_from_node(node1.node_id)
    dag.to_yaml("_pipelines/random_dag.yaml")

    dag2 = PipelineDAG(dag_file_path="_pipelines/random_dag.yaml")
    dag2.run_from_node(node1.node_id)
    dag2.to_yaml("_pipelines/random_dag_2.yaml")

if __name__ == "__main__":
    main()
