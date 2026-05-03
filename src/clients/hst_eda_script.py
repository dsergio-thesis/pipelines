
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    label_def_file = config.label_def_file
    max_records = config.max_records
    print("Configuration loaded successfully.")

    dag = PipelineDAG(label=pipeline_name)
    dag_dir = dag.dag_dir

    hst_catalog = HSTNodeCatalog(
            parameters={"max_records": max_records}, 
            dag_dir=dag_dir) 
    hst_eda = HSTNodeEDA(
            parameters={"max_records": max_records},
            dag_dir=dag_dir,
            parents=[hst_catalog.node_id])
    hst_clean = HSTNodeClean(
            parameters={"max_records": max_records},
            dag_dir=dag_dir,
            parents=[hst_eda.node_id])
    hst_clean_eda = HSTNodeEDA(
            parameters={"max_records": max_records},
            dag_dir=dag_dir,
            parents=[hst_clean.node_id])
    hst_select_clean = HSTNodeSelect(
            parameters={"max_records": max_records},
            dag_dir=dag_dir,
            parents=[hst_clean_eda.node_id])
    hst_select_eda = HSTNodeEDA(
            parameters={"max_records": max_records},
            dag_dir=dag_dir,
            parents=[hst_select_clean.node_id])

    dag.add_node(hst_catalog)
    dag.add_node(hst_eda)
    dag.add_node(hst_clean)
    dag.add_node(hst_clean_eda)
    dag.add_node(hst_select_clean)
    dag.add_node(hst_select_eda)
    
    dag.run_from_node(hst_catalog.node_id)
    dag.to_yaml()
    dag.to_graphviz()


if __name__ == "__main__":
    main()
