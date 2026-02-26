
import importlib
import sys
import os
from astroos_pipelines.config.astroos_config import AstroosConfig
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])

def test_config_defaults():
     
    # source env.sh
    env_file = 'env.sh'
    with open(env_file) as f:
        for line in f:
            if line.startswith('export'):
                key, value = line.replace('export', '').strip().split('=')
                os.environ[key] = value
                
    config = AstroosConfig.from_cli()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    pipeline_name = config.pipeline_name
    pipeline_minor_version = config.pipeline_minor_version
    max_records = config.max_records
    label_def_file = config.label_def_file

    assert os.path.isdir(os.path.join(dataset_dir, dataset_name)), f"Dataset directory {os.path.join(dataset_dir, dataset_name)} does not exist."
    assert pipeline_name is not None 
    assert pipeline_minor_version == 1 
    assert max_records == 3
    assert os.path.isfile(os.path.join(label_def_file)), f"Label definition file {os.path.join(dataset_dir, dataset_name, label_def_file)} does not exist." 
    




