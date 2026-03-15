
from __future__ import annotations
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

class Node(ABC):
    """Represents a node in the DAG, which can be a task or an operation."""

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Node.registry[cls.__name__] = cls

    def __init__(self, 
                 node_type, 
                 node_id=None,
                 parents=[],
                 parameters=None, 
                 inputs=[], 
                 outputs=[]):
        self.node_type = node_type
        self.parents = parents
        self.children = []
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs
        self.visited = False
        
        if node_id:
            self.node_id = node_id
        else:
            self.node_id = compute_node_id(
                node_type=self.node_type,
                parent_ids=[p for p in parents],
                params=self.parameters,
                # artifact_hashes=[a.output_path for a in self.inputs + self.outputs],
            )

        print(f"creating a node with parents {parents}")
        
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
            print(f"inputs: {d.get('inputs')}")
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])]
        if d.get("outputs") is not None and len(d.get("outputs")) > 0:
            print(f"outputs: {d.get('outputs')}")
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])]

        parent_ids=d.get("parent_ids", []),

        subclass = cls.registry[node_type]
        return subclass._from_dict(d)


    def output_fits_table(self, table: Table):
        file_path = os.path.join(
                "_pipelines", 
                self.node_id, 
                f"{self.node_type}.fits")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        table.write(file_path, format="fits", overwrite=True)

        artifact = Artifact(
            name=self.node_type,
            file_path=file_path
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
    file_path: str = None

    def to_dict(self):
        return {
            "name": self.name,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            file_path=d["file_path"],
        )


def compute_node_id(node_type, parent_ids, params, artifact_hashes=[]):
    payload = {
        "node_type": node_type,
        "parents": sorted(parent_ids),
        "params": params,
        "artifacts": sorted(artifact_hashes),
    }

    # deterministic serialization
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    # sha256 hash
    node_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return node_id[:8]  # shorten for readability


class PipelineDAG(DAG):
    """DAG Pipeline"""

    def __init__(self, dag_file_path=None):
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
            yaml.dump(
                {
                    "nodes": [i.to_dict() for i in self.nodes.values()]
                },
                file,
                sort_keys=False
            )

    def run_from_node(self, node_id):
        node = self.nodes[node_id]
        for _, node in self.nodes.items():
            node.visited = False
        
        self._run_node(node_id)
    
    def _run_node(self, node_id):
        print(f"Running node {node_id} with children {self.children[node_id]}")
        node = self.nodes[node_id]
        
        for p in node.parents:
            parent_node = self.nodes[p]
            print(f"parent_node.outputs: {parent_node.outputs}")
            node.inputs.extend(parent_node.outputs)

        node.run()
        
        if node_id in self.children:
            for child_id in self.children[node_id]:
                self._run_node(child_id)
                
        node.visited = True


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

        np.random.seed(42)
        n = 100
        data = Table({
            "objectId": np.arange(n),
            "feature1": np.random.rand(n),
            "feature2": np.random.rand(n),
            "label": np.random.randint(0, 2, n),
        })
        self.output_fits_table(data)
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

        np.random.seed(42)
        n = 100
        data = Table({
            "objectId": np.arange(n),
            "feature1": np.random.rand(n),
            "feature2": np.random.rand(n),
            "label": np.random.randint(0, 2, n),
        })

        self.output_fits_table(data)
        print(f"Transformed random data with {len(data)} rows.")


