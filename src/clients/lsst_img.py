
from clients._rsp import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    # dataset_cart_cutouts_morph = FITS_Image_Morphometry_Photometry_Dataset(
            # dataset_dir=os.path.join(dataset_dir, dataset_name),
            # labels_init_file=label_def_file,
            # N_bands=5, 
            # N_morphometric_features=4,
            # N_photometric_features=4,
            # )

    dataset_cart_cutouts_morph_b = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            )

    catalog = LSSTNodeCatalog(
            parameters={
                "max_records": max_records,
                "query_coords": pipeline_metadata.get("query_coords", None),
                "query_radius": pipeline_metadata.get("query_radius", None),})

    match = LSSTNodeMatchToHST(
            parameters={
                "max_records": max_records,},
            parents=[catalog.node_id],
            )

    preprocess = LSSTNodePreprocess(
            parameters={
                "dataset": dataset_cart_cutouts_morph_b.to_dict()
                },
            parents=[match.node_id],
            )

    fetch = LSSTNodeButlerFetch(
            parameters={
                "dataset": dataset_cart_cutouts_morph_b.to_dict()
                },
            parents=[preprocess.node_id],
            )

    dag = PipelineDAG()

    dag.add_node(catalog)
    dag.add_node(match)
    dag.add_node(preprocess)
    dag.add_node(fetch)

    dag.run_from_node(catalog.node_id)

    dag.to_yaml("_pipelines/lsst_img_pipeline.yaml")

    # pipelines = [
            # PipelineClassification(
                # name=pipeline_name,
                # metadata=pipeline_metadata,
                # max_records=max_records,
                # dataset=dataset_cart_cutouts_morph,
                # minor_version=None,
                # ),
            # PipelineClassification(
                # name=pipeline_name,
                # metadata=pipeline_metadata,
                # max_records=max_records,
                # dataset=dataset_cart_cutouts_morph_b,
                # minor_version=None,
                # ),
            # ]

    # pipelines[0].add_stages([ # Soda
        # StageCatalogLSST(),
        # StageMatchLSSTtoHST(),
        # StagePreprocessLSST(),
        # StageFetchLSSTSoda(),
        # ])

    # pipelines[1].add_stages([ # Butler
        # StageCatalogLSST(),
        # StageMatchLSSTtoHST(),
        # StagePreprocessLSST(),
        # StageButlerFetchLSST(),
        # ])

    # pipelines[1].run_pipeline()

if __name__ == "__main__":
    main()
