
from clients._rsp import *

def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    parameter = config.parameter
    option_create = config.option_create
    node_type = config.node_type
    node_label = config.node_label
    max_records = config.max_records
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    labels_def_file = config.labels_def_file

    target = config.sky_region_target_selected

    # dataset_cart_phot = FITS_Image_Morphometry_Photometry_Dataset(
            # dataset_dir=os.path.join(dataset_dir, dataset_name),
            # labels_init_file=labels_def_file,
            # )

    print(f"Label: {node_label}")
    
    dag = PipelineDAG(label=pipeline_name)
    if not dag.is_initialized():
        print("No pipelines found.")
        PipelineDAG.usage()
        return

    dag_dir = dag.dag_dir

    if option_create:
        if node_type == "tap":
            if not target:
                print("--target required to run LSST pipeline")
                return

            coords = config.sky_region_targets[target].sky_coord
            radius = config.sky_region_targets[target].radius_arcmin
            dag_node = NodeTAPQuery(label=node_label,
                        parameters={
                            "max_records": max_records,
                            "query_coords": coords,
                            "query_radius": radius,
                            "script": "catalogs/collections/lsst-hst/lsst/scripts/query.py",
                            },
                        origin=True
                        )
        elif node_type == "script":
            dag_node = NodeScript(label=node_label)
        elif node_type == "import":
            dag_node = NodeImport(label=node_label, parameters={"max_records": max_records})
        elif node_type == "export":
            dag_node = NodeExport(label=node_label)
        elif node_type == "eda":
            dag_node = NodeEDA(label=node_label)
        elif node_type == "eda-script":
            dag_node = NodeEDAScript(label=node_label)
        else: 
            dag_node = NodeGeneric(label=node_label)

        dag.add_node(dag_node)

    if input_artifact:
        dag.add_input_artifact_item(input_artifact)
    if parameter:
        dag.add_parameter(parameter)

    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()
