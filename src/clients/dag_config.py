
from clients._local import *

def main():

    config = client_config()
    target = config.sky_region_target_selected
    max_records = config.max_records

    if target:
        config.set_target(target)

    if max_records:
        config.set_env_var("MAX_RECORDS", max_records)

    print(config)

if __name__ == "__main__":
    main()

