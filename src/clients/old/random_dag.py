
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
            parameters={'max_records': max_records, 'max_value': 1},
            )

    node2 = NodeTransformRandom(
            node_type='transform_random', 
            parents=[node1.node_id],
            parameters={'max_records': max_records, 'multiplier': 20},
            )

    node3 = NodeTransformRandom(
            node_type='transform_random', 
            parents=[node1.node_id],
            parameters={'max_records': max_records, 'multiplier': 30},
            )

    nodeA = NodeCatalogRandom(
            node_type='catalog_random', 
            parameters={'max_records': max_records, 'max_value': 5},
            )

    nodeB = NodeTransformRandom(
            node_type='transform_random', 
            parents=[nodeA.node_id],
            parameters={'max_records': max_records, 'multiplier': 50},
            )
    
    node4 = NodeMergeRandom(
            node_type='transform_merge_random', 
            parents=[node2.node_id, node3.node_id, nodeB.node_id],
            parameters={'max_records': max_records},
            )

    dag = PipelineDAG()
    dag.add_node(node1)
    dag.add_node(node2)
    dag.add_node(node3)

    # Adding a separate branch to test multiple roots and merges
    dag.add_node(nodeA)
    dag.add_node(nodeB)

    # Adding node4 which merges outputs from node2, node3, and nodeB to test multiple parents
    dag.add_node(node4)

    dag.to_graphviz()
            
    print(f"Running DAG with node: {node1.node_id}")
    dag.run_from_node(node1.node_id)
    dag.to_yaml("_pipelines/random_dag.yaml")

    # dag2 = PipelineDAG(dag_file_path="_pipelines/random_dag.yaml")
    # dag2.run_from_node(node1.node_id)
    # dag2.to_yaml("_pipelines/random_dag_2.yaml")

if __name__ == "__main__":
    main()
