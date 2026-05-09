
from clients._rsp import *

def main():

    config = client_config()
    dag_dir = config.pipeline_dir

    max_records = config.max_records
    target = config.sky_region_target_selected

    if not target:
        print("--target required to run LSST pipeline")
        return

    coords = config.sky_region_targets[target].sky_coord
    radius = config.sky_region_targets[target].radius_arcmin

    print(f"Target: {target}, Radius (arcmin): {radius}, SkyCoord: ")
    print(coords)

    lsst_catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": coords,
                "query_radius": radius,
                })
    lsst_export = NodeExport(
            parents = [lsst_catalog.node_id],
            )
    lsst_eda = NodeEDA(
            parents = [lsst_export.node_id], 
            )
    # lsst_preprocess = LSSTNodePreprocess(
            # parameters={"max_records": max_records},
            # )
    # lsst_preprocess_eda = LSSTNodeEDA(
            # parameters={"max_records": max_records,},
            # )

    dag = PipelineDAG(new=True)

    dag.add_node(lsst_catalog, new_artifact = True)
    dag.add_node(lsst_export)
    dag.add_node(lsst_eda)
    
    # dag.add_node(lsst_preprocess)
    # dag.add_node(lsst_preprocess_eda)

    dag.run()
    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
