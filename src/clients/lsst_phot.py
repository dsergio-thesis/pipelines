
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    label_def_file = config.label_def_file
    max_records = config.max_records
    print("Configuration loaded successfully.")

    dataset_cart_phot = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            )
    
    dag = PipelineDAG(label=pipeline_name)
    dag_dir = dag.dag_dir

    lsst_catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": pipeline_metadata.get("query_coords", None),
                "query_radius": pipeline_metadata.get("query_radius", None),},
            dag_dir=dag_dir,
            origin=True)


    # we usually want the whole catalog for matching. adjust just for testing this
    hst_max_records = 300000
    hst_catalog = HSTNodeCatalog(
            parameters={"max_records": hst_max_records},
            dag_dir=dag_dir,
            origin=True) 
    hst_clean = HSTNodeClean(
            parameters={"max_records": hst_max_records},
            parents=[hst_catalog.node_id],
            dag_dir=dag_dir)
    hst_select_clean = HSTNodeSelect(
            parameters={"max_records": hst_max_records},
            parents=[hst_clean.node_id],
            dag_dir=dag_dir)

    lsst_hst_match = LSSTNodeMatchToHST(
            parameters={
                "max_records": max_records,},
            parents=[lsst_catalog.node_id, hst_select_clean.node_id],
            dag_dir=dag_dir,
            )

    lsst_hst_preprocess = LSSTNodePreprocess(
            parameters={
                "max_records": max_records,},
            parents=[lsst_hst_match.node_id],
            dag_dir=dag_dir
            )
    lsst_hst_data = LSSTNodePhotoDataset(
            parameters={
                "dataset": dataset_cart_phot.to_dict()},
            parents=[lsst_hst_preprocess.node_id],
            dag_dir=dag_dir
            )


    dag.add_node(hst_catalog)
    dag.add_node(hst_clean)
    dag.add_node(hst_select_clean)

    dag.add_node(lsst_catalog)
    dag.add_node(lsst_hst_match)
    dag.add_node(lsst_hst_preprocess)
    dag.add_node(lsst_hst_data)

    # dag.run_from_node(lsst_catalog.node_id)
    dag.run()

    dag.to_graphviz()

    dag.to_yaml()

if __name__ == "__main__":
    main()
