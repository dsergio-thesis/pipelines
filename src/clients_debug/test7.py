
from astroos_pipelines.artifacts import *
import pandas as pd
from pprint import pprint

def main():

    dag = ArtifactDAG()

    dag.add_edge("n1", "n2")
    dag.add_edge("n2", "n3")
    dag.add_edge("n2", "n4")
    dag.add_edge("n3", "n5")
    dag.add_edge("n4", "n6")
    dag.add_edge("n5", "n7")
    dag.add_edge("n6", "n7")
    dag.add_edge("n7", "n8")

    file_name = "test-a1.csv"
    with open(file_name, "w") as f:
        f.write("")


    a1 = ArtifactItem(file_name, dag)


    a1 = ArtifactItem("test-a1.csv", dag)

    a1.add_column("A", "n1", [1, 2, 3, 4, 5])
    a1.add_column("B", "n1", ["a", "b", "c", "d", "e"])

    a1.add_column("C", "n2", [True, False, True, False, True])

    a1.add_column("A", "n5", [10, 20, 30, 40, 50])
    a1.add_column("A", "n6", [0.1, 0.2, 0.3, 0.4, 0.5])

    print(a1.to_df("n5"))

    a1.to_csv("n5")

    pprint(a1.columns)


    
if __name__ == "__main__":
    main()
