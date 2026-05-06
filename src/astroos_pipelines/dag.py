
from __future__ import annotations
from graphviz import Digraph
from dataclasses import dataclass
import hashlib
import json
from typing import dataclass_transform
import yaml
from abc import ABC, abstractmethod
from collections import defaultdict
import importlib
import sys
import os
import numpy as np
from astropy.table import Table
from pathlib import Path
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
import inspect

from astroos_pipelines.utils.formatting import ascii_kv_table

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])

def to_plain_data(obj):
    if isinstance(obj, SkyCoord):
        icrs = obj.icrs
        return {
            "type": "SkyCoord",
            "frame": "icrs",
            "ra_deg": float(icrs.ra.deg),
            "dec_deg": float(icrs.dec.deg),
        }
    if isinstance(obj, Quantity):
        return {
            "type": "Quantity",
            "value": float(obj.value),
            "unit": str(obj.unit),
        }
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_plain_data(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_data(v) for v in obj]
    return obj


class Node(ABC):
    """Represents a node in the DAG, which can be a task or an operation."""

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Node.registry[cls.__name__] = cls

    def __init__(self, 
                 node_type, 
                 dag_dir,
                 label=None,
                 description=None,
                 node_id=None,
                 parents=[],
                 parameters={}, 
                 inputs=[], 
                 outputs=[]):
        self.dag_dir = dag_dir
        self.node_type = node_type
        self.label = label or node_type
        self.description = description or "Default node description."
        self.parents = parents
        self.children = []
        self.parameters = parameters
        self.inputs = []
        self.outputs = []
        self.visited = False
        
        if node_id:
            self.node_id = node_id
        else:
            self.node_id = compute_node_id(
                node_type=self.node_type,
                parent_ids=[p for p in parents],
                params=self.parameters,
                artifact_hashes=[a.output_path for a in self.inputs + self.outputs],
            )
        self.node_dir = os.path.join(dag_dir, self.node_id)

        # print(f"Creating {self.node_id} with parents {parents}")
        
    @abstractmethod
    def run(self):
        pass

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "parents": [p for p in self.parents],
            "node_type": self.node_type,
            "dag_dir": self.dag_dir,
            "node_dir": self.node_dir,
            "parameters": self.parameters,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [i.to_dict() for i in self.outputs],
        }
    @classmethod
    def from_dict(cls, d):
        # print("Getting node from dict")
        # print(f"Node dict: {d}")
        node_type = d["type"]
        node_id=d["node_id"]
        node_dir=d["node_dir"]
        dag_dir=d["dag_dir"]
        parameters=d.get("parameters", {})

        if d.get("inputs") is not None and len(d.get("inputs")) > 0:
            # print(f"inputs: {len(d.get('inputs'))}")
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])]
        if d.get("outputs") is not None and len(d.get("outputs")) > 0:
            # print(f"outputs: {len(d.get('outputs'))}")
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])]

        parent_ids=d.get("parents", [])

        subclass = cls.registry[node_type]
        # print(f"Found subclass {subclass} d: {d}")
        ret = subclass._from_dict(d)
        ret.inputs = inputs if 'inputs' in locals() else []
        ret.outputs = outputs if 'outputs' in locals() else []
        ret.parents = parent_ids if 'parent_ids' in locals() else []
        ret.node_id = node_id
        ret.node_dir = node_dir
        ret.dag_dir = dag_dir
        # print(f"Created node from dict: {ret}")
        return ret


    def output_fits_table(self, table: Table, columns=None):

        file_path = os.path.join(
                self.node_dir, 
                f"{self.node_type}.fits")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table.write(file_path, format="fits", overwrite=True)

        artifact = Artifact(
            name=self.node_type,
            file_path=file_path,
            columns=columns,
        )

        self.outputs = [artifact]
    
    def output_csv_table(self, table: Table, columns=None):

        # generate a unique random id for the file name
        file_id = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        file_path = os.path.join(
                self.node_dir,
                f"{self.node_type}.csv")
        
        # print(f"output_csv_table. dag_dir: {self.dag_dir}, node_dir: {self.node_dir}, file_path: {file_path}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table.write(file_path, format="csv", overwrite=True)

        artifact = Artifact(
            name=self.node_type,
            file_path=file_path,
            columns=columns if columns else None,
        )

        self.outputs = [artifact]

    def __repr__(self):
        rows = [("node_id", self.node_id), 
                ("type", self.node_type),
                ("parents", self.parents),
                ("children", self.children),
                ("parameters", self.parameters), 
                ]
        for i, input in enumerate(self.inputs):
            rows.append((f"input_{i}", input.file_path))
        for i, output in enumerate(self.outputs):
            rows.append((f"output_{i}", output.file_path))
        
        return ascii_kv_table(rows, title=f"Node ({self.node_id})")

    
    def node_label(self):
        return f"""
<table border="0" cellborder="0" cellspacing="0">
<tr>
<td>
<font point-size="18"><b><u>{self.label}</u></b></font>
</td>
</tr>
<tr>
<td>
<font >#{self.node_id}</font>
</td>
</tr>
<tr><td><font >{self.description}</font></td></tr>
<tr><td align="left"><font >• {len(self.inputs)} inputs ⇾ {len(self.outputs)} outputs</font></td></tr>
</table>
            """
        

class DAG(ABC):
    @abstractmethod
    def add_node(self, node: Node, parents=[]):
        pass

    @abstractmethod
    def get_nodes(self):
        pass

    @abstractmethod
    def to_yaml(self, file_path: str):
        pass

    @abstractmethod
    def get_nodes_ids(self):
        pass

    @abstractmethod
    def run_from_node(self, node_id):
        pass

class Artifact:
    """DAG artifact, the data that flows between nodes in the DAG."""

    def __init__(self,
                 name: str,
                 file_path: str,
                 columns: dict = None,
                 ):
        self.name = name
        self.file_path = file_path
        self.columns = columns

        if columns is None:
            # print(f"Columns not provided for artifact {self.name}, trying to get from file path {self.file_path}")
            self.try_get_columns()  # attempt to populate columns if not provided

    def try_get_columns(self):
        # # print(f"Trying to get columns for artifact {self.name} from file {self.file_path}")
        if self.file_path is not None and os.path.exists(self.file_path):
            try:
                table = Table.read(self.file_path)
                self.columns = {col: str(table[col].dtype) for col in table.colnames}
            except Exception as e:
                # print(f"Error reading file {self.file_path}: {e}")
                pass

    def to_dict(self):
        return {
            "name": self.name,
            "file_path": self.file_path,
            "columns": self.columns,
        }

    def __eq__(self, other):
        if not isinstance(other, Artifact):
            return False
        return self.name == other.name and self.file_path == other.file_path 

    @classmethod
    def from_dict(cls, d):
        # print(f"Creating Artifact from dict: {d}")
        return cls(
            name=d["name"],
            file_path=d["file_path"],
            columns=d.get("columns", None),
        )

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Quantity

def make_jsonable(obj):
    if isinstance(obj, SkyCoord):
        return {
            "__type__": "SkyCoord",
            "ra_deg": round(obj.ra.deg, 12),
            "dec_deg": round(obj.dec.deg, 12),
            "frame": obj.frame.name,
        }
    if isinstance(obj, Quantity):
        return {
            "__type__": "Quantity",
            "value": round(float(obj.value), 12),
            "unit": str(obj.unit),
        }
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    return obj

def compute_node_id(node_type, parent_ids, params, artifact_hashes=[]):
    payload = {
        "node_type": node_type,
        "parents": sorted(parent_ids),
        "params": make_jsonable(params),
        "artifacts": sorted(artifact_hashes),
    }

    # deterministic serialization
    serialized = json.dumps(payload, 
                            sort_keys=True, 
                            separators=(",", ":"))

    # sha256 hash
    node_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    # for now set to random hash
    hash = hashlib.sha256(os.urandom(16)).hexdigest()
    node_id = hash

    return node_id[:8]  # shorten for readability


class PipelineDAG(DAG):
    """DAG Pipeline"""

    def __init__(self, 
                 label=None):

        if not label:
            if not os.path.exists(os.path.join("_pipelines", "dags_index.yaml")):
                with open(os.path.join("_pipelines", "dags_index.yaml"), "w") as file:
                    yaml.safe_dump({"dags": [], "selected_dag": None}, file, sort_keys=False)

            with open(os.path.join("_pipelines", "dags_index.yaml"), "r") as file:
                index_data = yaml.safe_load(file)
                selected_dag = index_data.get("selected_dag")
                if selected_dag:
                    label = selected_dag
                else:
                    # set to random hash if no label provided and there is no selected DAG in the index
                    label = hashlib.sha256(os.urandom(16)).hexdigest()[:8]

        # set dag_id to label lower case and replace any non-alphanumeric characters with underscores
        self.label = label
        self.dag_id = "".join(c if c.isalnum() else "_" for c in label.lower())
        self.dag_dir = os.path.join("_pipelines", self.dag_id)

        # create directory for the DAG if it doesn't exist
        os.makedirs("_pipelines", exist_ok=True)
        os.makedirs(self.dag_dir, exist_ok=True)

        dag_exists = os.path.exists(os.path.join(self.dag_dir, "dag.yaml"))
        self.dag_file_path = os.path.join(self.dag_dir, "dag.yaml")

        self.head = None

        if dag_exists:
            # print("DAG file found, loading DAG from file.")
            with open(self.dag_file_path, "r") as file:
                data = yaml.safe_load(file)
                # print(f"Loaded DAG data: {data}")
                self.nodes = {}
                        # node_data["node_id"]: Node.from_dict(node_data) for node_data in data["nodes"]}

                for node_data in data["nodes"]:
                    # print()
                    # print(f"- Node data: {node_data}")
                    node = Node.from_dict(node_data)
                    # print(f" ** Created node: {node}")
                    # print()
                    self.nodes[node_data["node_id"]] = node

                # print(f"Loaded DAG with nodes: {self.nodes}, type: {type(self.nodes)}")
                self.children = defaultdict(list)
                for node in self.nodes.values():
                    # print(f"Node {node}, num inputs: {len(node.inputs)}, num outputs: {len(node.outputs)}, parents: {node.parents}")
                    for parent in node.parents:
                        self.children[parent].append(node.node_id)
                if data["head"] is not None:
                    # print(f"Head node: {data['head']}")
                    self.head = self.nodes[data["head"]]

        else:
            # print("No DAG file found, initializing new DAG.")
            self.nodes = {} 
            self.children = defaultdict(list)

        dags_index = os.path.join("_pipelines", "dags_index.yaml")
        index_data = {
                "dags": [], 
                "selected_dag": self.dag_id, 
                }
        if os.path.exists(dags_index):
            with open(dags_index, "r") as file:
                index_data = yaml.safe_load(file)
                if self.dag_id not in index_data["dags"]:
                    index_data["dags"].append(self.dag_id)
                    index_data["selected_dag"] = self.dag_id
                    with open(dags_index, "w") as file:
                        yaml.safe_dump(index_data, file, sort_keys=False)
        else:
            with open(dags_index, "w") as file:
                index_data["dags"].append(self.dag_id)
                yaml.safe_dump(index_data, file, sort_keys=False)
        
        # print(f"Initialized DAG with nodes: {self.nodes}, type: {type(self.nodes)}")
        # print(f"Initialized DAG Index with DAGs: {index_data['dags'] if os.path.exists(dags_index) else [self.dag_id]}")

    def add_node(self, node: Node):
        node_id = node.node_id
        for p in node.parents: 
            if p not in self.get_nodes_ids():
                raise ValueError("Parent must exist")
            self.children[p].append(node_id)

        # if dag is empty and no parents provided, set parents to empty, else if no parents provided, set parents to [head]
        if len(self.nodes) == 0 and len(node.parents) == 0:
            node.parents = []
        elif len(node.parents) == 0:
            node.parents = [self.head.node_id] if self.head else []

        print(f"Adding node {node_id} with parents {[p for p in node.parents]}")
        print(repr(node))

        self.nodes[node_id] = node
        self.head = node
        self.to_yaml()  # save DAG after adding node
    
    def get_head(self):
        """ Get the head node of the DAG"""
        # print(f"Getting head node: {self.head}")
        return self.head

    def print_head(self):
        if self.head:
            # print(f"Head node: {self.head.node_id}, type: {self.head.node_type}, parameters: {self.head.parameters}")
            # print(f"Head node inputs: {[i.file_path for i in self.head.inputs]}")
            # print(f"Head node outputs: {[o.file_path for o in self.head.outputs]}")
            pass
        else:
            # print("No head node set.")
            pass

    def add_input_artifact(self, file_path):
        artifact = Artifact(
            name=os.path.basename(file_path),
            file_path=file_path,
        )
        # print(f"Head node: {self.get_head()}")
        # print(f"existing inputs: {[i.file_path for i in self.head.inputs]}")
        # print(f"Adding input artifact {artifact} to head node {self.head.node_id if self.head else None}")

        self.get_head().inputs = [artifact]

    def get_nodes_ids(self):
        ids = []
        for node_id, node in self.nodes.items():
            ids.append(node_id)
        return ids

    def get_nodes(self):
        return list(self.nodes)

    def to_yaml(self, file_path=None):

        if file_path is None:
            file_path = os.path.join(self.dag_dir, "dag.yaml")
        with open(file_path, "w") as file:

            data = {
                "head": self.head.node_id if self.head else None,
                "nodes": [node.to_dict() for node in self.nodes.values()]
            }
            yaml.safe_dump(to_plain_data(data), file, sort_keys=False)

        # also write each node to a separate yaml file for easier debugging
        # for node_id, node in self.nodes.items():
            # node_file_path = os.path.join(self.dag_dir, f"{node_id}.yaml")
            # with open(node_file_path, "w") as file:
                # yaml.safe_dump(to_plain_data(node.to_dict()), file, sort_keys=False)
    

    def to_graphviz(self, view=False):
        # dot = Digraph(comment=self.label)
        dot = Digraph()
        # set title with extra padding around it
        # dot.attr(label=f"{self.label}\n ", labelloc="t", fontsize="20")

        dot.attr(rankdir="LR")  # left to right

        # top to bottom: rankdir="TB"
        dot.attr("node", 
                 shape="box", 
                 style="filled,rounded", 
                 fillcolor="#D6A095",
                 fontcolor="#2E2E2E",
                 color="#8F3F2B",
                 penwidth="2",
                 bgcolor="transparent")
        dot.graph_attr.update(bgcolor="transparent")
        dot.attr(
            "edge",
            penwidth="2.5",
            arrowsize="1.2",
            color="#64748b"
        )

        for node_id, node in self.nodes.items():
            # label = f"{node.node_type}\n{node_id[:8]}"
            label = node.node_label() 
            dot.node(node_id, f"<{label}>")

        for node_id, node in self.nodes.items():
            for parent_id in node.parents:
                dot.edge(parent_id, node_id)

        output_path = os.path.join(self.dag_dir, "dag")
        dot.render(output_path, format="png", cleanup=True, view=view)
        dot.save(os.path.join(self.dag_dir, "dag.dot"))
        return dot

    def run_from_node(self, node_id):
        node = self.nodes[node_id]
        for _, node in self.nodes.items():
            node.visited = False
        
        self._run_node(node_id)
    
    def _run_node(self, node_id):

        # make dir
        os.makedirs(self.nodes[node_id].node_dir, exist_ok=True)

        node = self.nodes[node_id]
        print(repr(node))

        node.visited = True
        
        for p in node.parents:
            parent_node = self.nodes[p]

            if not parent_node.visited:
                self._run_node(p)
            # print(f"{parent_node.node_id} outputs: {len(parent_node.outputs)}")

            for output in parent_node.outputs:
                if output not in node.inputs:
                    node.inputs.append(output)

        node.run()
        source_code = inspect.getsource(node.run)
        source_code_path = os.path.join(node.node_dir, f"{node.node_id}-script.py")
        with open(source_code_path, "w") as f:
            f.write(source_code)

        # print(f"Finished running node {node_id}. parameters: {node.parameters}, inputs: {[i.file_path for i in node.inputs]}, outputs: {[o.file_path for o in node.outputs]}")
        if node.parameters and "last_run_source" in node.parameters:
            self.nodes[node_id].parameters["last_run_source"] = source_code_path
        
        if node_id in self.children:
            for child_id in self.children[node_id]:
                child_node = self.nodes[child_id]
                if not child_node.visited:
                    self._run_node(child_id)

    def run(self):
        if self.head is None:
            # print("No head node set, cannot run DAG.")
            return
        self.run_from_node(self.head.node_id)
        
    @staticmethod
    def list_dags():
        index_file = os.path.join("_pipelines", "dags_index.yaml")
        if os.path.exists(index_file):
            with open(index_file, "r") as file:
                index_data = yaml.safe_load(file)
                selected_dag = index_data.get("selected_dag")
                # print(f"Available DAGs: {index_data['dags']}")
                # print(f"Selected DAG: {selected_dag}")


class NodeGeneric(Node):
    """
    A generic node that can be used for testing the pipeline.
    """

    def __init__(self,
                 dag_dir,
                 node_type="generic",
                 node_id=None,
                 parents=[],
                 parameters={},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )


    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeGeneric"
        return d

    @classmethod
    def _from_dict(cls, d):
        # print(f"Creating NodeGeneric from dict: {d}")

        return cls(
            node_id=d["node_id"],
            dag_dir = d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )


    def run(self):
        """ Pass through inputs to outputs for testing the pipeline. """
        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            data = Table.read(artifact.file_path)
            self.output_csv_table(data)
            # print(f"Passed through data with {len(data)} rows and {len(data.colnames)} columns.")


class NodeCatalogRandom(Node):
    """
    Generate random data for testing the pipeline.
    """

    def __init__(self,
                 node_type="catalog_random",
                 node_id=None,
                 parents=[],
                 parameters={},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeCatalogRandom"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        n = self.parameters.get("n", 10)
        max_value = self.parameters.get("max_value", 10)

        np.random.seed(0)
        data = Table({
            "objectId": np.arange(n),
            "feature1": np.random.rand(n) * max_value,
            "feature2": np.random.rand(n) * max_value,
            "label": np.random.randint(0, 2, n),
        })

        self.output_csv_table(data)
        # print(f"Generated random data with {len(data)} rows.")


class NodeTransformRandom(Node):
    """
    Apply random transformations to the data for testing the pipeline.
    """

    def __init__(self,
                 node_type="transform_random",
                 node_id=None,
                 parents=[],
                 parameters={},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeTransformRandom"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            data = Table.read(artifact.file_path)

            columns = artifact.columns if artifact.columns else data.colnames
            for col in columns:
                if col in data.colnames and np.issubdtype(data[col].dtype, np.number):
                    data[col] = data[col] * self.parameters.get("multiplier", 2)

            self.output_csv_table(data, columns=columns)
            # print(f"Transformed data with multiplier {self.parameters.get('multiplier', 2)}.")


class NodeMergeRandom(Node):
    """
    Merge for testing the pipeline.
    """

    def __init__(self,
                 node_type="merge_random",
                 node_id=None,
                 parents=[],
                 parameters={},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeMergeRandom"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        # expects two input artifacts
        if len(self.inputs) < 2:
            raise ValueError(f"NodeMergeRandom expects at least two input artifacts, got {len(self.inputs)}")
        artifact1 = self.inputs[0]
        artifact2 = self.inputs[1]
        data1 = Table.read(artifact1.file_path)
        data2 = Table.read(artifact2.file_path)

        columns1 = artifact1.columns if artifact1.columns else data1.colnames
        columns2 = artifact2.columns if artifact2.columns else data2.colnames

        merged = data1[columns1].copy()
        for col in columns2:
            if col in merged.colnames:
                merged[col + "_2"] = data2[col]
            else:
                merged[col] = data2[col]

        self.output_csv_table(merged)
        # print(f"Merged data with {len(merged)} rows and {len(merged.colnames)} columns.")


class NodeBadToNaN(Node):
    """
    A node that introduces NaN values for testing the pipeline's handling of missing data.
    """

    def __init__(self,
                 dag_dir,
                 node_type="bad_to_nan",
                 node_id=None,
                 parents=[],
                 parameters={},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )
        if self.parameters is None:
            self.parameters = {
                "bad_map": {
                    "Av": [-1],
                    "z_best": [-99],
                }
            }

    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeBadToNaN"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir=d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            columns = artifact.columns if artifact.columns else None
            data = Table.read(artifact.file_path)
            df = data.to_pandas()
            # print(df)

            bad_map = self.parameters.get("bad_map", {})

            for col, bad_vals in bad_map.items():
                if col in df.columns:
                    df[col] = df[col].replace(bad_vals, np.nan) # replace bad values with nan
            
            # print(df)
            data = Table.from_pandas(df)
            # print(f"Replaced bad values with NaN in columns {list(bad_map.keys())}.")
            self.output_csv_table(data, columns=columns)


class NodeScript(Node):
    """
    A node that runs a user-provided script for testing the pipeline's ability to run arbitrary code.

    """

    def __init__(self,
                 dag_dir,
                 node_type="script",
                 node_id=None,
                 parents=[],
                 parameters={"script": None},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
        )

        if self.parameters['script'] is None:
            # write template script to node directory
            template_script = """# Example script for NodeScript
# This script will be executed when the node runs. You can access input artifacts, parameters, and output artifacts to perform custom operations.
"""         
            script_path = os.path.join(self.node_dir, f"{node_id}-script.py")

            os.makedirs(self.node_dir, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(template_script)
            self.parameters = {
                "script": script_path
                }
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeScript"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir=d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            columns = artifact.columns if artifact.columns else None
            data = Table.read(artifact.file_path)
            df = data.to_pandas()
            # print(df)

            if self.parameters is not None and "script" in self.parameters:
                script = self.parameters.get("script", "")

                with open(script, "r") as f:
                    code = f.read()
                    exec(code, {"df": df, "parameters": self.parameters, "inputs": self.inputs, "outputs": self.outputs})

            # print(f"Executed script {script} on data with {len(df)} rows and {len(df.columns)} columns.")
            # print(df)

            data = Table.from_pandas(df)
            self.output_csv_table(data, columns=columns)
