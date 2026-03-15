
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")
    
    # does not work
    mast = AstroosQueryMast(root_dir = "test_data", res_object_identifier_column="obsid") 
    res = mast.query(query_params=params)
    print(res)
    print(res.columns)


if __name__ == "__main__":
    main()
