from __future__ import annotations

import pprint
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

import html
import textwrap
from graphviz import Digraph

from astroos_pipelines.utils.formatting import ascii_kv_table
from astroos_pipelines.utils.plots.dataset_eda import *
from astroos_pipelines.artifacts import *

# from astroos_pipelines.hst.dag import *

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.artifacts'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
# importlib.reload(sys.modules['astroos_pipelines.hst.dag'])

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
                 outputs=[],
                 origin=False,
                 num_inputs=0,
                 num_outputs=0,
                 ):
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
        self.diff = {}
        self.artifact_dag = None 
        self.origin = origin
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        
        if node_id:
            self.node_id = node_id
        else:
            self.node_id = compute_node_id(
                node_type=self.node_type,
                parent_ids=[p for p in parents],
                params=self.parameters,
                artifact_hashes=[a.output_path for a in self.inputs + self.outputs],
            )

        print(f"setting dag_dir in constructor")
        self.set_dag_dir(dag_dir)

        # print(f"Creating {self.node_id} with parents {parents}")

    def node_configure(self):
        """Configure the node before running. This can be used to set default parameters or perform setup tasks."""
        pass
    
    def set_dag_dir(self, dag_dir = None):
        self.dag_dir = dag_dir
        if dag_dir is not None:
            self.node_dir = os.path.join(dag_dir, self.node_id)
        else:
            print(f"dag_dir is None. Cannot set self.node_dir for {self.node_id}")

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
            "diff": self.diff,
        }
    @classmethod
    def from_dict(cls, d):
        # print("Getting node from dict")
        # print(f"Node dict: {d}")
        node_type = d["type"]
        node_id=d["node_id"]
        node_dir=d["node_dir"]
        dag_dir=d["dag_dir"]
        diff=d.get("diff", {})
        parameters=d.get("parameters", {})

        if d.get("inputs") is not None and len(d.get("inputs")) > 0:
            # print(f"inputs: {len(d.get('inputs'))}")
            # inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])]
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])]
        if d.get("outputs") is not None and len(d.get("outputs")) > 0:
            # print(f"outputs: {len(d.get('outputs'))}")
            # outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])]
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])]

        parent_ids=d.get("parents", [])

        if node_type not in Node.registry:
            raise ValueError(
                f"Unknown node type {node_type}. "
                f"Registered types: {list(cls.registry.keys())}"
                )

        # subclass = cls.registry[node_type]
        subclass = Node.registry[node_type]
        # print(f"Found subclass {subclass} d: {d}")
        ret = subclass._from_dict(d)
        ret.inputs = inputs if 'inputs' in locals() else []
        ret.outputs = outputs if 'outputs' in locals() else []
        ret.parents = parent_ids if 'parent_ids' in locals() else []
        ret.node_id = node_id
        ret.node_dir = node_dir
        ret.dag_dir = dag_dir
        ret.parameters = parameters
        ret.diff = diff
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

    def to_yaml_string(self):
        return yaml.safe_dump(to_plain_data(self.to_dict()), sort_keys=False)

    def yaml_to_html_label(self, yaml_text: str, width_chars: int = 80, width_px: int = 320) -> str:
        html_lines = []

        for line in yaml_text.splitlines():
            indent_len = len(line) - len(line.lstrip(" "))
            indent = "&nbsp;" * indent_len
            content = line[indent_len:]

            wrapped = textwrap.wrap(
                content,
                width=max(1, width_chars - indent_len),
                break_long_words=True,
                break_on_hyphens=False,
            )

            if not wrapped:
                html_lines.append(indent)
            else:
                html_lines.append(indent + html.escape(wrapped[0]))

                # continuation lines preserve same indentation
                for extra in wrapped[1:]:
                    html_lines.append(indent + html.escape(extra))
        return "<br align='left'/>".join(html_lines)
    
    def node_label(self):
        yaml_html = self.yaml_to_html_label(self.to_yaml_string())

        # wrap description to fit within the node width
        # desc_wrapped = textwrap.fill(self.description, width=40)
        
        # then convert to <br align='left'/> for html label
        desc_html = self.yaml_to_html_label(self.description, width_chars=40)
        

        return f"""
<table border="0" cellborder="0" cellspacing="0">
<tr>
<td>
<font point-size="18"><b><u>{self.label}</u></b></font>
</td>
</tr>
<tr>
<td>
<font>#{self.node_id}</font>
</td>
</tr>
<tr><td align="left">
<br align="left"/>{desc_html}
<br align="left"/>
<br align="left"/><font point-size="8">{yaml_html}</font>
<br align="left"/>
<br align="left"/>
<br align="left"/> • {len(self.inputs)} inputs ⇾ {len(self.outputs)} outputs
</td></tr>
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
                 artifact_dag: ArtifactDAG = None,
                 label: str = None,
                 new: bool = False,
                 ):

        self.head = None
        self.dag_id = None
        self.dag_dir = None
        self.artifact_dag = artifact_dag
        os.makedirs("_pipelines", exist_ok=True)
        self.nodes = {} 
        self.children = defaultdict(list)

        if not label:
            if not os.path.exists(os.path.join("_pipelines", "dags_index.yaml")):
                with open(os.path.join("_pipelines", "dags_index.yaml"), "w") as file:
                    yaml.safe_dump({"dags": [], "selected_dag": None}, file, sort_keys=False)

            with open(os.path.join("_pipelines", "dags_index.yaml"), "r") as file:
                index_data = yaml.safe_load(file)
                selected_dag = index_data.get("selected_dag")
                if selected_dag and not new:
                    label = selected_dag
                elif new:
                    # set to random hash if no label provided and there is no selected DAG in the index
                    label = hashlib.sha256(os.urandom(16)).hexdigest()[:8]

        if not label:
            return

        # set dag_id to label lower case and replace any non-alphanumeric characters with underscores
        self.label = label
        self.dag_id = "".join(c if c.isalnum() else "_" for c in label.lower())
        self.dag_dir = os.path.join("_pipelines", self.dag_id)

        # create directory for the DAG if it doesn't exist
        os.makedirs(self.dag_dir, exist_ok=True)

        dag_exists = os.path.exists(os.path.join(self.dag_dir, "dag.yaml"))
        self.dag_file_path = os.path.join(self.dag_dir, "dag.yaml")

        if dag_exists:
            with open(self.dag_file_path, "r") as file:
                data = yaml.safe_load(file)
                self.nodes = {}

                for node_data in data["nodes"]:
                    node = Node.from_dict(node_data)
                    self.nodes[node_data["node_id"]] = node

                artifact_dag_data = data["artifact_dag"]
                if artifact_dag_data is not None:
                    self.artifact_dag = ArtifactDAG.from_dict(artifact_dag_data)
                else:
                    self.artifact_dag = ArtifactDAG()

                self.children = defaultdict(list)
                for node in self.nodes.values():
                    for parent in node.parents:
                        self.children[parent].append(node.node_id)
                if data["head"] is not None:
                    self.head = self.nodes[data["head"]]
        else:
            print("No pipelines found, initializing new PipelineDAG...")
            self.artifact_dag = ArtifactDAG()

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
        
    def is_initialized(self):
        return self.dag_id is not None
        
    def add_node(self, node: Node, new_artifact: bool = False):

        node.set_dag_dir(self.dag_dir)

        node.node_configure()

        node_id = node.node_id
        print(f"Adding node {node_id}")
        node.artifact_dag = self.artifact_dag
       
        head = self.head

        # if dag is empty and no parents provided, set parents to empty, else if no parents provided, set parents to [head]
        if len(self.nodes) == 0 and len(node.parents) == 0:
            node.parents = []
            self.head = node
        elif len(node.parents) == 0 and not node.origin:
            node.parents = [self.head.node_id] if self.head else []
        elif node.origin:
            node.parents = []
            self.head = node

        head = self.head

        for p in node.parents: 
            if p not in self.get_nodes_ids():
                raise ValueError(f"Parent must exist p={p}, self.get_nodes_ids(): {self.get_nodes_ids()}. Hint: you need to set parents in Node() constructor if running in a script. ")
            self.children[p].append(node_id)
            self.artifact_dag.add_edge(p, node_id)

        if new_artifact:
            artifact_item = ArtifactItem(
                file_path = os.path.join(self.dag_dir, node.node_id, "catalog.fits"),
                dag=self.artifact_dag,
                node_id=head.node_id if head else None,
            )
            if head.inputs is None:
                head.inputs = []
            head.inputs.append(artifact_item)

        print(f"Adding node {node_id} of type {node.node_type} with parents {[p for p in node.parents]}")
        print(repr(node))

        # make dir for the node
        os.makedirs(node.node_dir, exist_ok=True)
        print(f"Node {node_id} directory created at: ", node.node_dir)

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

    def add_input_artifact_item(self, file_path):
        print(f"Adding input artifact item with file path {file_path} to head node {self.head.node_id if self.head else None}")
        head = self.get_head()

        if (not head or head.node_type != "import"):
            raise ValueError("Can only add artifact to head node of type 'import'")
        artifact_item = ArtifactItem(
            file_path=file_path,
            dag=self.artifact_dag,
            node_id=head.node_id if head else None,
        )
        if head.inputs is None:
            head.inputs = []
        head.inputs.append(artifact_item)
        
    def add_input_artifact(self, file_path):
        # old Artifact way, keeping for reference

        artifact = Artifact(
            name=os.path.basename(file_path),
            file_path=file_path,
        )
        # print(f"Head node: {self.get_head()}")
        # print(f"existing inputs: {[i.file_path for i in self.head.inputs]}")
        # print(f"Adding input artifact {artifact} to head node {self.head.node_id if self.head else None}")

        self.get_head().inputs = [artifact]

    def add_parameter(self, parameter):
        print(f"Adding parameter {parameter} to head node {self.head.node_id if self.head else None}")

        print(f"{len(parameter)} type: {type(parameter)}, existing parameters: {self.get_head().parameters if self.head else None}")

        value = parameter[1]
        # if numeric string like "10" or "3.14", convert to int or float
        if isinstance(value, str):
            if value.isdigit():
                value = int(value)
            elif value in ["true", "True", "false", "False"]:
                value = value.lower() == "true"
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        
        self.get_head().parameters[parameter[0]] = value

    def get_nodes_ids(self):
        ids = []
        for node_id, node in self.nodes.items():
            ids.append(node_id)
        return ids

    def get_nodes(self):
        return list(self.nodes)

    def to_yaml(self, file_path=None):

        print(f"Saving DAG to yaml. artifact_dag: {self.artifact_dag}")

        if file_path is None:
            file_path = os.path.join(self.dag_dir, "dag.yaml")
        with open(file_path, "w") as file:

            for node in self.nodes.values():
                node.diff = Artifact.prune_empty(node.diff)

            data = {
                "head": self.head.node_id if self.head else None,
                "nodes": [node.to_dict() for node in self.nodes.values()],
                "artifact_dag": self.artifact_dag.to_dict() if self.artifact_dag else None,
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

        dot.attr(rankdir="TB")  # left to right (LR) or top to bottom (TB)

        dot.attr("node", 
                 shape="box", 
                 style="filled,rounded", 
                 fillcolor="#D6A095",
                 fontcolor="#2E2E2E",
                 color="#8F3F2B",
                 penwidth="2",
                 bgcolor="transparent",
                 width="6",
                 )
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

        node = self.nodes[node_id]
        node.artifact_dag = self.artifact_dag

        node.visited = True
        
        for p in node.parents:
            parent_node = self.nodes[p]

            if not parent_node.visited:
                self._run_node(p)
            # print(f"{parent_node.node_id} outputs: {len(parent_node.outputs)}")

            for output in parent_node.outputs:
                if output not in node.inputs:
                    node.inputs.append(output)

        # before = Artifact.snapshot(node.inputs, node.outputs)

        print(f"Running node {node_id}")
        print(repr(node))
        node.run()

        # for artifact in node.inputs + node.outputs:
            # artifact.refresh()
        # after = Artifact.snapshot(node.inputs, node.outputs)
        # changes = Artifact.diff_snapshots(before, after)
        # node.diff = changes
        # print()
        # print(f"Changes after running node {node_id}:")
        # pprint(changes)

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
    def usage():
        print("Usage:")
        print("  rad init [Optional -n <name>] - Initialize a new pipeline with an optional name.")

    def status(self):
        index_file = os.path.join("_pipelines", "dags_index.yaml")
        if os.path.exists(index_file):
            with open(index_file, "r") as file:
                index_data = yaml.safe_load(file)
                selected_dag = index_data.get("selected_dag")
                
                if len(index_data["dags"]) == 0:
                    print("No pipelines found.")
                    PipelineDAG.usage()
                    return

                print()
                print(f"Pipelines:")
                for dag in index_data["dags"]:
                    if dag == selected_dag:
                        print(f" * {dag}")
                    else:
                        print(f"   {dag}")
                print()

                if self.head:
                    print(f"Current head node: {self.head.node_id} of type {self.head.node_type}")

class NodeImport(Node):
    """
    Import Node. 
    """
    def __init__(self,
                 dag_dir=None,
                 node_type="import",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
                 inputs=[],
                 outputs=[],
                 origin=False,
                 ):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            origin=origin,
            description="Import data from external source into the pipeline.",
        )


    def node_configure(self):
        if not self.parameters or "script" not in self.parameters:
            # write template script to node directory
            template_script = """# Example script for NodeImport.

# print(f"*** Running in NodeImport...")
active_columns.update({  
    'ra': "Right Ascension",
    'dec': "Declination",
    'jh_mag': "J-H color",
    'z_spec': "Spectroscopic redshift",
    'z_peak_phot': "Photometric redshift (peak)",
    'z_peak_grism': "Grism redshift (peak)",
    'z_best': "Best redshift",
    'sfr': "Star Formation Rate",
    'lssfr': "Log Specific Star Formation Rate",
    'sfr_IR': "Star Formation Rate from IR",
    'sfr_UV': "Star Formation Rate from UV",
    'lmass': "Log Stellar Mass",
    'Av': "Visual Extinction",
    'beta': "UV slope",
    'L_IR': "Infrared Luminosity",
    'chi2': "Chi-squared of SED fit",
})

"""         
            script_path = os.path.join(self.node_dir, f"script.py")

            os.makedirs(self.node_dir, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(template_script)
            self.parameters = {
                "script": script_path
                }

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeImport"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir = d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        """ Import artifact. """
        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            
            print(f"Running NodeImport with artifact {artifact.file_path} and parameters {self.parameters}")

            active_columns = {}

            script = self.parameters.get("script", "")
            with open(script, "r") as f:
                code = f.read()
                exec(code, {"parameters": self.parameters, "inputs": self.inputs, "outputs": self.outputs, "active_columns": active_columns})
            
            print(f"Active columns after running script: {active_columns}")

            if self.parameters is not None and "max_records" in self.parameters:
                max_records = self.parameters["max_records"]
            else:
                max_records = 10

            if active_columns:
                print(f"Selected columns from script: {active_columns}")
                artifact.set_active_columns(active_columns)
            else:
                print("No active columns selected in script, using all columns.")
            
            artifact.dag = self.artifact_dag
            node_dir = os.path.join(self.dag_dir, self.node_id)
            artifact.file_path = os.path.join(node_dir, os.path.basename(artifact.file_path))

            print(f"Importing artifact {artifact.file_path} with dag {self.artifact_dag} and node_id {self.node_id}")

            if not os.path.exists(artifact.file_path):
                os.makedirs(node_dir, exist_ok=True)
                artifact.materialize(node_id=self.node_id, max_records=max_records)

            self.outputs = [artifact]


class NodeExport(Node):
    """
    Export Node
    """
    def __init__(self,
                 dag_dir=None,
                 node_type="export",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
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
            description="Export data by materializing the input artifacts to their file paths.",
        )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeExport"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir = d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        """ Export Node inputs to node_dir as full files. """
        if len(self.inputs) > 0:

            if self.outputs is None:
                self.outputs = []
            for artifact in self.inputs:
                print(f"Exporting artifact {artifact.file_path} with dag {self.artifact_dag} and node_id {self.node_id}")
                artifact.dag = self.artifact_dag
                artifact.file_path = os.path.join(self.dag_dir, self.node_id, os.path.basename(artifact.file_path))
                artifact.materialize(node_id=self.node_id)
                self.outputs.append(artifact)




class NodeGeneric(Node):
    """
    Generic dummy node.
    """

    def __init__(self,
                 dag_dir=None,
                 node_type="generic",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
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
            description="A generic node that can be used for testing the pipeline. It simply passes through the input artifacts to the output without modification.",
        )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeGeneric"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir = d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        """ Pass through inputs to outputs for testing the pipeline. """
        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            data = Table.read(artifact.file_path)
            
            df = data.to_pandas()

            self.outputs = [artifact]



class NodeEDA(Node):
    def __init__(
            self,
            dag_dir=None,
            node_type="eda",
            node_id=None,
            parents=[],
            parameters=None,
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label="EDA",
            description="Exploratory Data Analysis",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = self.__class__.__name__
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            dag_dir=d["dag_dir"],
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        print(f"running EDA {self.inputs}")

        if len(self.inputs) == 0:
            # print("No input artifact for EDA node.")
            return

        artifact = self.inputs.pop()
        print(f"Running EDA on artifact {artifact.file_path}")

        ext = os.path.splitext(artifact.file_path)[1].lower()
        if ext == ".fits":
            table = Table.read(artifact.file_path, hdu=1, format="fits")
        elif ext == ".csv":
            table = Table.read(artifact.file_path, format="csv")
        else:
            raise ValueError(f"Unsupported file format {ext} for EDA node.")

        columns = artifact.active_columns 

        if self.parameters is not None and "max_records" in self.parameters:
            max_records = self.parameters["max_records"]
            table = table[:max_records]

        dataset_eda(table=table, 
                    columns=columns, 
                    save_dir=self.node_dir, 
                    title="Exploratory Data Analysis",
                    )
        
        self.outputs = [artifact]
        # self.output_fits_table(table, columns=columns) # pass through the table to the next node


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
                 dag_dir=None,
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
            description="A node that runs a user-provided script. The script should be a Python file that can access input artifacts, parameters, and output artifacts.",
        )
    
    def node_configure(self):
        if self.parameters['script'] is None:
            # write template script to node directory
            template_script = """# Example script for NodeScript

# bad_map = {
    # "Av": [-1],
    # "L_IR": [-99],
    # "beta": [-99],
    # "chi2": [-1],
    # "sfr": [-99],
    # "sfr_IR": [-99],
    # "sfr_UV": [-99],
    # "z_best": [-99],
    # "z_peak_grism": [-1],
    # "z_peak_phot": [-99],
    # "z_spec": [-99.9],
# }
# for col, bad_vals in bad_map.items():
    # if col in df.columns:
        # df[col] = df[col].replace(bad_vals, np.nan) # replace bad values with nan

"""         
            script_path = os.path.join(self.node_dir, f"script.py")

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
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        if len(self.inputs) > 0:
            artifact = self.inputs[0] # expects one input artifact
            columns = artifact.columns if artifact.columns else None
            data = Table.read(artifact.file_path)
            df = data.to_pandas()
            # print(df)

            columns = artifact.active_columns if artifact.active_columns else data.colnames

            script = self.parameters.get("script", "")

            with open(script, "r") as f:
                code = f.read()
                exec(code, {"df": df, "parameters": self.parameters, "inputs": self.inputs, "outputs": self.outputs, "columns": columns})

            # print(f"Executed script {script} on data with {len(df)} rows and {len(df.columns)} columns.")
            # print(df)

            self.outputs = [artifact] 



class NodeJoin(Node):
    """
    Join dataframes by celestial position.
    """

    def __init__(self,
                 dag_dir=None,
                 node_type="join",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
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
            description="Join two Dataframes by RA and DEC. Requires two input artifacts with columns 'ra' and 'dec'. The output artifact will be the result of a Astropy match. Requires minimum separation parameter.",
        )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeJoin"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir = d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        """ Join dataframes. """
        if len(self.inputs) == 2:
            artifact1 = self.inputs[0]
            artifact2 = self.inputs[1]

            data1 = Table.read(artifact1.file_path)
            data2 = Table.read(artifact2.file_path)
            df1 = data1.to_pandas()
            df2 = data2.to_pandas()

            print("df1", df1)
            print("df2", df2)

            return

            # check for variations on RA and DEC column names
            if "ra" not in df1.columns or "dec" not in df1.columns:
                ra_col1 = None
                dec_col1 = None
                for col in df1.columns:
                    if col.lower() in ["coord_ra", "ra_deg", "ra_j2000", "ra"]:
                        ra_col1 = col
                    if col.lower() in ["coord_dec", "dec_deg", "dec_j2000", "dec"]:
                        dec_col1 = col
                if ra_col1 and dec_col1:
                    df1 = df1.rename(columns={ra_col1: "ra", dec_col1: "dec"})
                else:
                    raise ValueError(f"Input artifact 1 must have columns 'ra' and 'dec' or variations thereof. Columns: {df1.columns}")
            if "ra" not in df2.columns or "dec" not in df2.columns:
                ra_col2 = None
                dec_col2 = None
                for col in df2.columns:
                    if col.lower() in ["coord_ra", "ra_deg", "ra_j2000", "ra"]:
                        ra_col2 = col
                    if col.lower() in ["coord_dec", "dec_deg", "dec_j2000", "dec"]:
                        dec_col2 = col
                if ra_col2 and dec_col2:
                    df2 = df2.rename(columns={ra_col2: "ra", dec_col2: "dec"})
                else:
                    raise ValueError(f"Input artifact 2 must have columns 'ra' and 'dec' or variations thereof. Columns: {df2.columns}")

            c1 = SkyCoord(df1[ra_col1].to_numpy() * u.deg,
                            df1[dec_col1].to_numpy() * u.deg)
            c2 = SkyCoord(df2[ra_col2].to_numpy() * u.deg,
                            df2[dec_col2].to_numpy() * u.deg)
            idx, d2d, _ = c1.match_to_catalog_sky(c2)
            sep_arcsec = sep2d.to(u.arcsec).value

            m = (sep_arcsec < self.parameters.get("max_sep_arcsec", 1))
            df_matched = df1[m].copy()
            df_matched["matched_idx"] = idx[m]
            df_matched["sep_arcsec"] = sep_arcsec[m]

            df1_columns = artifact1.active_columns if artifact1.active_columns else {}
            df2_columns = artifact2.active_columns if artifact2.active_columns else {}

            matched_columns = {}
            for col in df1_columns:
                matched_columns[col] = df1_columns[col]
            for col in df2_columns:
                if col in matched_columns:
                    matched_columns[col + "_2"] = df2_columns[col]
                else:
                    matched_columns[col] = df2_columns[col]

            output_artifact = ArtifactItem(
                path=os.path.join(self.dag_dir, self.node_id, "joined.fits"),
                dag=self.artifact_dag,
                node_id=self.node_id,
                active_columns=matched_columns,
            )

            self.outputs = [output_artifact]
