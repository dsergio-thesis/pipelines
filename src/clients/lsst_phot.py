
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    dataset_cart_phot = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            N_bands=6, 
            N_morphometric_features=0,
            N_photometric_features=4,
            )

    lsst_catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": pipeline_metadata.get("query_coords", None),
                "query_radius": pipeline_metadata.get("query_radius", None),})

    hst_max_records = 300000
    hst_catalog = HSTNodeCatalog(parameters={"max_records": hst_max_records}) 
    hst_clean = HSTNodeClean(parameters={"max_records": hst_max_records},
                     parents=[hst_catalog.node_id])
    hst_select_clean = HSTNodeSelect(parameters={"max_records": hst_max_records},
                     parents=[hst_clean.node_id])

    lsst_hst_match = LSSTNodeMatchToHST(
            parameters={
                "max_records": max_records,},
            parents=[lsst_catalog.node_id, hst_select_clean.node_id],
            )

    lsst_hst_preprocess = LSSTNodePreprocess(
            parameters={
                "max_records": max_records,},
            parents=[lsst_hst_match.node_id],
            )
    lsst_hst_data = LSSTNodePhotoDataset(
            parameters={
                "dataset": dataset_cart_phot.to_dict()},
            parents=[lsst_hst_preprocess.node_id],
            )

    dag = PipelineDAG()

    dag.add_node(hst_catalog)
    dag.add_node(hst_clean)
    dag.add_node(hst_select_clean)

    dag.add_node(lsst_catalog)
    dag.add_node(lsst_hst_match)
    dag.add_node(lsst_hst_preprocess)
    dag.add_node(lsst_hst_data)

    dag.run_from_node(lsst_catalog.node_id)

    dag.to_graphviz()

    dag.to_yaml("_pipelines/lsst_phot_pipeline.yaml")

if __name__ == "__main__":
    main()
