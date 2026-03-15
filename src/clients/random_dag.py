
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    node1 = Node(
            stage=StageCatalogRandom(),
            node_type='catalog_random', 
            parameters={'max_records': max_records},
            outputs=[Artifact(name='catalog_random')]
            )

    node2 = Node(
            stage=StageTransformRandom(),
            node_type='transform_random', 
            parents=[node1],
            parameters={'max_records': max_records},
            outputs=[Artifact(name='transform_random')]
            )
    dag = PipelineDAG()
    dag.add_node(node1)
    dag.add_node(node2)
            
    print(f"Running DAG with node: {node1.node_id}")
    dag.run(node1.node_id)
    dag.to_yaml("random_dag.yaml")

    dag2 = PipelineDAG(dag_file_path="random_dag.yaml")
    dag2.to_yaml("random_dag_2.yaml")
    

    # pipelines = [
            # PipelineClassification(
                # name=pipeline_name,
                # metadata=pipeline_metadata,
                # max_records=max_records,
                # dataset=dataset_cart_cutouts_morph,
                # minor_version=None,
                # ),
            # ]

    # pipelines[0].add_stages([ 
        # StageCatalogRandom(),
        # StageTransformRandom(),
        # ])


    # for p in pipelines:
        # p.run_pipeline()

if __name__ == "__main__":
    main()
