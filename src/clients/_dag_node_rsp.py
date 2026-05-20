
from clients._rsp import *

# Below is copied from dag_node.py 
def main():

    config = client_config()
    pipeline_name = config.pipeline_name
    input_artifact = config.input_artifact
    parameter = config.parameter
    option_create = config.option_create
    option_origin = config.option_origin
    parent = config.parent
    node_type = config.node_type
    node_label = config.node_label
    max_records = config.max_records
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    labels_def_file = config.labels_def_file

    target = config.sky_region_target_selected

    print(f"Label: {node_label}")
    
    dag = PipelineDAG(label=pipeline_name)
    if not dag.is_initialized():
        print("No pipelines found.")
        PipelineDAG.usage()
        return

    parent_id = dag.get_node_id(parent) if parent else None

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
        elif node_type == "eda-script":
            dag_node = NodeEDAScript(label=node_label)
        elif node_type == "merge":
            dag_node = NodeJoin(parameters={"max_sep_arcsec": 0.8,})
        elif node_type == "photo-dataset":
            dag_node = NodePhotometricDataset()
        else: 
            dag_node = NodeGeneric(label=node_label)
            if parent_id:
                dag_node.parents = [parent_id]

        dag.add_node(dag_node)
        if option_origin:
            dag_node.parents = []
    
    if parent_id:
        dag.head.parents.append(parent_id)

    if input_artifact:
        dag.add_input_artifact_item(input_artifact)

    if parameter:
        dag.add_parameter(parameter)

    if parameter and parameter[0] == "dataset-name":
        dataset_name = parameter[1]
        dataset = FITS_Image_Morphometry_Photometry_Dataset(
                dataset_dir=os.path.join(dataset_dir, dataset_name),
                labels_init_file=labels_def_file,
                )
        dag.add_parameter("dataset", dataset.to_dict())

    dag.to_yaml()
    dag.to_graphviz()

if __name__ == "__main__":
    main()

