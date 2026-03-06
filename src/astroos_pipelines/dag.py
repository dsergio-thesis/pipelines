
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

from astroos_pipelines.pipelines import StagePipeline

importlib.reload(sys.modules['astroos_pipelines.pipelines'])


class Node:
    """Represents a node in the DAG, which can be a task or an operation."""

    def __init__(self, 
                 stage: StagePipeline,
                 node_type, 
                 parents=[],
                 parameters=None, 
                 inputs=[], 
                 outputs=[]):
        self.node_type = node_type
        self.parents = parents
        self.parameters = parameters
        self.inputs = inputs
        self.outputs = outputs
        self.visited = False
        self.stage = stage

        print(f"creating a node with parents {parents}")

        node_id = compute_node_id(
            node_type=self.node_type,
            parent_ids=[p.node_id for p in parents],
            params=self.parameters,
            # artifact_hashes=[a.output_path for a in self.inputs + self.outputs],
        )
        self.node_id = node_id

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "parents": [p.node_id for p in self.parents],
            "node_type": self.node_type,
            "parameters": self.parameters,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "stage": self.stage.to_dict(),
        }

    def __str__(self):
        return (
            f"Node(id={self.node_id}, "
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
    def run(self, node_id):
        pass

@dataclass
class Artifact:
    """DAG artifact, the data that flows between nodes in the DAG."""
    name: str
    file_path: str = None


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

    return node_id


class PipelineDAG(DAG):
    """DAG Pipeline"""

    def __init__(self, dag_file_path=None):
        if dag_file_path:
            with open(dag_file_path, "r") as file:
                data = yaml.safe_load(file)
                self.nodes = {node_data["node_id"]: Node(**node_data) for node_data in data["nodes"]}
                self.children = defaultdict(list)
                for node in self.nodes.values():
                    for parent in node.parents:
                        self.children[parent.node_id].append(node.node_id)
        else:
            self.nodes = {} 
            self.children = defaultdict(list)

    def add_node(self, node: Node):
        node_id = node.node_id
        parents = node.parents
        print(f"Adding node {node_id} with parents {[p.node_id for p in parents]}")
        for p in parents:
            if p.node_id not in self.get_nodes_ids():
                raise ValueError("Parent must exist")

        self.nodes[node_id] = node
        self.children[node_id] = []
        self.children[node_id].extend(parents)

        node.pipeline_dag = self

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
                    "nodes": [node.to_dict() for node in self.nodes.values()]
                },
                file,
                sort_keys=False
            )

    def run(self, node_id):
        node = self.nodes[node_id]
        for _, node in self.nodes.items():
            node.visited = False
        
        self._run_node(node_id)
    
    def _run_node(self, node_id):
        print(f"Running node {node_id}")
        node = self.nodes[node_id]
        
        for p in node.parents:
            parent_node = self.nodes[p.node_id]

        stage = node.stage

        stage.run()
        node.visited = True

        if node_id in self.children:
            for child_id in self.children[node_id]:
                self._run_node(child_id)



