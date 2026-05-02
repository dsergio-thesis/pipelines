
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
                 label=None,
                 description=None,
                 node_id=None,
                 parents=[],
                 parameters=None, 
                 inputs=[], 
                 outputs=[]):
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

        print(f"Creating {self.node_id} with parents {parents}")
        
    @abstractmethod
    def run(self):
        pass

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "parents": [p for p in self.parents],
            "node_type": self.node_type,
            "parameters": self.parameters,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [i.to_dict() for i in self.outputs],
        }
    @classmethod
    def from_dict(cls, d):
        node_type = d["type"]
        node_id=d["node_id"]
        # node_type=d["node_type"]
        parameters=d.get("parameters", {})

        if d.get("inputs") is not None and len(d.get("inputs")) > 0:
            print(f"inputs: {len(d.get('inputs'))}")
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])]
        if d.get("outputs") is not None and len(d.get("outputs")) > 0:
            print(f"outputs: {len(d.get('outputs'))}")
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])]

        parent_ids=d.get("parent_ids", []),

        subclass = cls.registry[node_type]
        return subclass._from_dict(d)


    def output_fits_table(self, table: Table, columns=None):

        # generate a unique random id for the file name
        file_id = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        file_path = os.path.join(
                "_pipelines", 
                self.node_id,
                self.node_type,
                f"{file_id}.fits")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table.write(file_path, format="fits", overwrite=True)

        artifact = Artifact(
            name=self.node_type,
            file_path=file_path,
            columns=columns if columns else None,
        )

        if (self.outputs is None):
            self.outputs = [artifact]
        else:
            self.outputs.append(artifact)
    
    def output_csv_table(self, table: Table, columns=None):

        # generate a unique random id for the file name
        file_id = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        file_path = os.path.join(
                "_pipelines", 
                self.node_id,
                self.node_type,
                f"{file_id}.csv")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table.write(file_path, format="csv", overwrite=True)

        artifact = Artifact(
            name=self.node_type,
            file_path=file_path,
            columns=columns if columns else None,
        )

        if (self.outputs is None):
            self.outputs = [artifact]
        else:
            self.outputs.append(artifact)

    def __str__(self):
        return (
            f"Node(node_id={self.node_id}, "
            f"type={self.node_type}, "
            f"parameters={self.parameters}, "
            f"inputs={self.inputs}, outputs={self.outputs})"
            )
    
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

@dataclass
class Artifact:
    """DAG artifact, the data that flows between nodes in the DAG."""
    name: str
    columns: dict = None
    file_path: str = None

    def to_dict(self):
        return {
            "name": self.name,
            "file_path": self.file_path,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, d):
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

    return node_id[:8]  # shorten for readability


class PipelineDAG(DAG):
    """DAG Pipeline"""

    def __init__(self, 
                 dag_file_path=None,
                 label="Pipeline DAG"):

        # set dag_id to label lower case and replace any non-alphanumeric characters with underscores
        self.label = label
        self.dag_id = "".join(c if c.isalnum() else "_" for c in label.lower())

        if dag_file_path:
            with open(dag_file_path, "r") as file:
                data = yaml.safe_load(file)
                self.nodes = {
                        node_data["node_id"]: Node.from_dict(node_data) for node_data in data["nodes"]}
                print(f"Loaded DAG with nodes: {self.nodes}, type: {type(self.nodes)}")
                self.children = defaultdict(list)
                for node in self.nodes.values():
                    print(f"Node {node}")
                    for parent in node.parents:
                        self.children[parent].append(node.node_id)
        else:
            self.nodes = {} 
            self.children = defaultdict(list)

    def add_node(self, node: Node):
        node_id = node.node_id
        print(f"Adding node {node_id} with parents {[p for p in node.parents]}")
        for p in node.parents: 
            if p not in self.get_nodes_ids():
                raise ValueError("Parent must exist")
            self.children[p].append(node_id)

        self.nodes[node_id] = node
        

    def get_nodes_ids(self):
        ids = []
        for node_id, node in self.nodes.items():
            ids.append(node_id)
        return ids

    def get_nodes(self):
        return list(self.nodes)

    def to_yaml(self, file_path):
        with open(file_path, "w") as file:
            data = {
                "nodes": [node.to_dict() for node in self.nodes.values()]
            }
            yaml.safe_dump(to_plain_data(data), file, sort_keys=False)
    

    def to_graphviz(self, filename=None, view=False):
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

        if filename is None:
            filename = f"dag_{self.dag_id}"
        output_path = os.path.join("_pipelines", filename)
        dot.render(output_path, format="png", cleanup=True, view=view)
        dot.save(f"dag_{self.dag_id}.dot")
        return dot

    def run_from_node(self, node_id):
        node = self.nodes[node_id]
        for _, node in self.nodes.items():
            node.visited = False
        
        self._run_node(node_id)
    
    def _run_node(self, node_id):
        print(f"Running node {node_id} with children {self.children[node_id]}")
        node = self.nodes[node_id]
        node.visited = True
        
        for p in node.parents:
            parent_node = self.nodes[p]

            if not parent_node.visited:
                self._run_node(p)
            print(f"{parent_node.node_id} outputs: {len(parent_node.outputs)}")
            node.inputs.extend(parent_node.outputs)

        node.run()
        
        if node_id in self.children:
            for child_id in self.children[node_id]:
                child_node = self.nodes[child_id]
                if not child_node.visited:
                    self._run_node(child_id)
                


class NodeCatalogRandom(Node):
    """
    Generate random data for testing the pipeline.
    """

    def __init__(self,
                 node_type="catalog_random",
                 node_id=None,
                 parents=[],
                 parameters=None,
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
        print(f"Generated random data with {len(data)} rows.")


class NodeTransformRandom(Node):
    """
    Apply random transformations to the data for testing the pipeline.
    """

    def __init__(self,
                 node_type="transform_random",
                 node_id=None,
                 parents=[],
                 parameters=None,
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

        artifact = self.inputs[0] # expects one input artifact
        data = Table.read(artifact.file_path)

        columns = artifact.columns if artifact.columns else data.colnames
        for col in columns:
            if col in data.colnames and np.issubdtype(data[col].dtype, np.number):
                data[col] = data[col] * self.parameters.get("multiplier", 2)

        self.output_csv_table(data, columns=columns)
        print(f"Transformed data with multiplier {self.parameters.get('multiplier', 2)}.")


class NodeMergeRandom(Node):
    """
    Merge for testing the pipeline.
    """

    def __init__(self,
                 node_type="merge_random",
                 node_id=None,
                 parents=[],
                 parameters=None,
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
        print(f"Merged data with {len(merged)} rows and {len(merged.colnames)} columns.")


