
from clients._rsp import *

def main():

    config = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    labels_def_file = config.labels_def_file
    max_records = config.max_records

    target = config.sky_region_target_selected

    if not target:
        print("--target required to run LSST pipeline")
        return

    coords = config.sky_region_targets[target].sky_coord
    radius = config.sky_region_targets[target].radius_arcmin

    dataset_cart_phot = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=labels_def_file,
            )
    
    new_dag = True if pipeline_name is None else False
    dag = PipelineDAG(label=pipeline_name, new=new_dag)
    dag_dir = dag.dag_dir

    print(f"dag_dir={dag_dir}, Target: {target}, Radius (arcmin): {radius}, SkyCoord: ")
    print(coords)

    lsst_catalog = NodeTAPQuery(
            label="LSST DP-1 TAP",
            parameters={
                "max_records": max_records,
                "query_coords": coords,
                "query_radius": radius,
                "script": "catalogs/collections/lsst-hst/lsst/scripts/query.py",
                },
            origin=True
            )
    lsst_export = NodeExport(
            parents = [lsst_catalog.node_id],
            )
    lsst_clean = NodeScript(
            parameters = {
                "script": "catalogs/collections/lsst-hst/lsst/scripts/clean.py"
                },
            parents = [lsst_export.node_id],
            )
    lsst_select = NodeScript(
            parameters = {
                "script": "catalogs/collections/lsst-hst/lsst/scripts/select.py"
                },
            parents = [lsst_clean.node_id],
            )
    lsst_select_export = NodeExport(
            parents = [lsst_select.node_id],
            )

    
    # we usually want the whole catalog for matching. adjust just for testing this
    hst_max_records = 300000
    hst_catalog = NodeImport(
            label="3D-HST Catalog",
            parameters = {
                "max_records": hst_max_records,
                "script": "catalogs/collections/lsst-hst/hst/scripts/import.py",
                },
            origin=True,
            )
    hst_clean = NodeScript(
            parameters = {
                "script": "catalogs/collections/lsst-hst/hst/scripts/clean.py"
                },
            parents = [hst_catalog.node_id],
            )
    hst_select = NodeScript(
            parameters = {
                "script": "catalogs/collections/lsst-hst/hst/scripts/select.py"
                },
            parents = [hst_clean.node_id],
            )
    hst_export = NodeExport(
            parents = [hst_select.node_id],
            )



    lsst_hst_match = NodeJoin(
            parameters={
                "max_sep_arcsec": 0.8,},
            parents=[lsst_select_export.node_id, hst_export.node_id],
            )
    lsst_hst_export = NodeExport(
            parents = [lsst_hst_match.node_id],
            )
    lsst_hst_eda = NodeEDAScript(
            parameters={
                "eda_type": "histogram"
                },
            parents = [lsst_hst_export.node_id],
            )
    lsst_hst_sky_dist_eda = NodeEDAScript(
            parameters={
                "eda_type": "sky-distribution"
                },
            parents = [lsst_hst_eda.node_id],
            )
    
    lsst_hst_dataset = NodePhotometricDataset(
            parameters={
                "dataset": dataset_cart_phot.to_dict()},
            parents=[lsst_hst_sky_dist_eda.node_id],
            )

    new_artifact_path="catalogs/collections/lsst-hst/hst/hst.fits"
    dag.add_node(hst_catalog)
    dag.add_input_artifact_item(new_artifact_path)

    dag.add_node(hst_clean)
    dag.add_node(hst_select)
    dag.add_node(hst_export)

    dag.add_node(lsst_catalog)
    dag.add_node(lsst_export)
    dag.add_node(lsst_clean)
    dag.add_node(lsst_select)
    dag.add_node(lsst_select_export)

    dag.add_node(lsst_hst_match)
    dag.add_node(lsst_hst_export)
    dag.add_node(lsst_hst_eda)
    dag.add_node(lsst_hst_sky_dist_eda)
    dag.add_node(lsst_hst_dataset)




    # lsst_hst_data = LSSTNodePhotoDataset(
            # parameters={
                # "dataset": dataset_cart_phot.to_dict()},
            # parents=[lsst_hst_preprocess.node_id],
            # dag_dir=dag_dir
            # )
    # dag.add_node(lsst_hst_data)



    dag.run()
    dag.to_graphviz()
    dag.to_yaml()

if __name__ == "__main__":
    main()
