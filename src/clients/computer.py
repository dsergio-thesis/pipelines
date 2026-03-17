
from clients._local import *

def main():

    dataset_dir = "data"
    dataset_name = "computer"
    labels_def_file = "catalogs/computer_labels.csv"
    max_records = 100

    dataset_computer = Computer_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            )

    computer_catalog = ComputerNodeCatalog(
            parameters={
                "max_records": max_records,
                "dataset": dataset_computer.to_dict(),
                },

    dag = PipelineDAG()

    dag.add_node(computer_catalog)

    dag.run_from_node(computer_catalog.node_id)

    dag.to_graphviz()

    dag.to_yaml("_pipelines/computer_dag.yaml")

if __name__ == "__main__":
    main()
