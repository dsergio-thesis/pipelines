
from clients._local import *


def main():

    config = client_config()
    selected_dag_id = config.selected_dag_id
    
    dags = {}
    dags_index_file = os.path.join("_pipelines", "dags_index.yaml")
    if os.path.exists(dags_index_file):
        with open(dags_index_file, "r") as file:
            dags = yaml.safe_load(file)

        if selected_dag_id in dags["dags"]:
            dags["selected_dag"] = selected_dag_id
            with open(dags_index_file, "w") as file:
                yaml.dump(dags, file)
            print(f"Selected DAG '{selected_dag_id}' has been updated in the index.")
        else:
            print(f"DAG ID '{selected_dag_id}' not found in the index.")
    else:
        print(f"DAG index file '{dags_index_file}' does not exist.")


if __name__ == "__main__":
    main()

