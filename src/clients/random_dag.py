
import sys
import os
import importlib

from astroos_pipelines.pipelines import PipelineClassification, \
        StageCatalogRandom, StageTransformRandom
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.config.astroos_config import AstroosConfig

from astroos_pipelines.dag import PipelineDAG, Node, Artifact

importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
importlib.reload(sys.modules['astroos_pipelines.dag'])

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)

def main():

    config = AstroosConfig.random_data()
    coord, radius = config.get_target("CDF_South")
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version
    max_records = config.max_records
    label_def_file = config.label_def_file

    print()
    print(config)

    pipeline_metadata = {'query_coords': coord, 'query_radius': radius}
    print(pipeline_metadata)

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
