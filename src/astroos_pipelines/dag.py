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

from tqdm import tqdm
from astropy.io import fits

from astropy import units as u

import html
import textwrap
from graphviz import Digraph

from astroos_pipelines.utils.formatting import ascii_kv_table
from astroos_pipelines.utils.plots.dataset_eda import *
from astroos_pipelines.utils.plots.eda_histogram import *
from astroos_pipelines.utils.plots.eda_sky_distribution import *
from astroos_pipelines.utils.plots.eda_color_color import *
from astroos_pipelines.utils.plots.eda_pairplot import *
from astroos_pipelines.artifacts import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset


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

        # # print(f"setting dag_dir in constructor")
        self.set_dag_dir(dag_dir)

        # # print(f"Creating {self.node_id} with parents {parents}")

    def node_configure(self):
        """Configure the node before running. This can be used to set default parameters or perform setup tasks."""
        pass
    
    def set_dag_dir(self, dag_dir = None):
        self.dag_dir = dag_dir
        if dag_dir is not None:
            self.node_dir = os.path.join(dag_dir, self.node_id)
        else:
            # print(f"dag_dir is None. Cannot set self.node_dir for {self.node_id}")
            pass

    @abstractmethod
    def run(self):
        pass

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "parents": [p for p in self.parents],
            "node_type": self.node_type,
            "label": self.label,
            "description": self.description,
            "dag_dir": self.dag_dir,
            "node_dir": self.node_dir,
            "parameters": self.parameters,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [i.to_dict() for i in self.outputs],
        }
    @classmethod
    def from_dict(cls, d):
        # # print("Getting node from dict")
        # # print(f"Node dict: {d}")
        node_type = d["type"]
        label = d["label"]
        description = d.get("description", "")
        node_id=d["node_id"]
        node_dir=d["node_dir"]
        dag_dir=d["dag_dir"]
        parameters=d.get("parameters", {})

        if d.get("inputs") is not None and len(d.get("inputs")) > 0:
            # # print(f"inputs: {len(d.get('inputs'))}")
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])]
        if d.get("outputs") is not None and len(d.get("outputs")) > 0:
            # # print(f"outputs: {len(d.get('outputs'))}")
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])]

        parent_ids=d.get("parents", [])

        if node_type not in Node.registry:
            raise ValueError(
                f"Unknown node type {node_type}. "
                f"Registered types: {list(cls.registry.keys())}"
                )

        # subclass = cls.registry[node_type]
        subclass = Node.registry[node_type]
        # # print(f"Found subclass {subclass} d: {d}")
        ret = subclass._from_dict(d)
        ret.inputs = inputs if 'inputs' in locals() else []
        ret.outputs = outputs if 'outputs' in locals() else []
        ret.parents = parent_ids if 'parent_ids' in locals() else []
        ret.node_id = node_id
        ret.label = label
        ret.description = description
        ret.node_dir = node_dir
        ret.dag_dir = dag_dir
        ret.parameters = parameters
        # # print(f"Created node from dict: {ret}")
        return ret

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

    def yaml_to_html_label(self, yaml_text: str, width_chars: int = 80, width_px: int = 400) -> str:
        html_lines = []

        for line in yaml_text.splitlines():
            line = line.rstrip()
            indent_len = len(line) - len(line.lstrip(" "))
            indent = "&nbsp;" * indent_len
            content = line[indent_len:]

            wrapped = textwrap.wrap(
                content,
                width=max(1, width_chars - indent_len),
                break_long_words=True,
                break_on_hyphens=False,
            )
            # # print(f"Wrapping line: '{content}' with indent {indent_len} and width {width_chars} -> wrapped: {wrapped}")

            if not wrapped:
                html_lines.append(indent)
            else:
                html_lines.append(indent + html.escape(wrapped[0]))

                # continuation lines preserve same indentation
                for extra in wrapped[1:]:
                    html_lines.append(html.escape(extra))

        return "<br align='left'/>".join(html_lines) + "<br align='left'/>"
    
    def node_label(self):
        yaml_html = self.yaml_to_html_label(self.to_yaml_string(), width_chars=60)
        desc_html = self.yaml_to_html_label(self.description, width_chars=60)

        # # print(f"desc_html: {desc_html}")

        return f"""
<table border="0" cellborder="1" cellspacing="0" cellpadding="10" color="#CBD5E1">
<tr>
<td bgcolor="#F8FAFC" align="center">
<font face="Helvetica" point-size="18" color="#0F172A"><b>{self.label}</b></font>
<br/>
<font face="Helvetica" point-size="10" color="#64748B">#{self.node_id}</font>
</td>
</tr>

<tr>
<td bgcolor="#FFFFFF" align="left">
<font face="Helvetica" point-size="11" color="#334155"><br align="left"/>{desc_html}</font>
</td>
</tr>

<tr>
<td bgcolor="#F1F5F9" align="left"><br align="left"/>
    <font face="Courier" point-size="8" color="#475569">
<br align="left"/>
{yaml_html}
    </font>
</td>
</tr>

<tr>
<td bgcolor="#E0F2FE" align="center">
<font face="Helvetica" point-size="10" color="#075985">{len(self.inputs)} inputs &#8594; {len(self.outputs)} outputs</font>
</td>
</tr>
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
            # print("No pipelines found, initializing new PipelineDAG...")
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
        
    def add_node(self, node: Node, new_artifact: bool = False, new_artifact_path: str = None):

        node.set_dag_dir(self.dag_dir)

        node.node_configure()

        node_id = node.node_id
        # print(f"Adding node {node_id}")
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
    
        if not new_artifact_path:
            new_artifact_path = os.path.join(node.node_dir, "catalog.fits")

        if new_artifact:
            artifact_item = ArtifactItem(
                file_path = new_artifact_path,
                dag=self.artifact_dag,
                node_id=head.node_id if head else None,
            )
            if head.inputs is None:
                head.inputs = []
            head.inputs.append(artifact_item)

        print(f"Adding node {node_id} of type {node.node_type} with parents {[p for p in node.parents]}")
        # print(repr(node))

        # make dir for the node
        os.makedirs(node.node_dir, exist_ok=True)
        # print(f"Node {node_id} directory created at: ", node.node_dir)

        self.nodes[node_id] = node
        self.head = node
        self.to_yaml()  # save DAG after adding node
    
    def get_head(self):
        """ Get the head node of the DAG"""
        return self.head

    def add_input_artifact_item(self, file_path):
        # print(f"Adding input artifact item with file path {file_path} to head node {self.head.node_id if self.head else None}")
        head = self.get_head()
        if (not head or head.node_type != "import"):
            # print(f"Current head node type: {head.node_type if head else None}")
            raise ValueError("Can only add artifact to head node of type 'import'")

        # first copy file as-is to node dir
        os.makedirs(head.node_dir, exist_ok=True)
        with open(file_path, "rb") as src_file:
            dest_path = os.path.join(head.node_dir, os.path.basename(file_path))
            with open(dest_path, "wb") as dest_file:
                dest_file.write(src_file.read())
        
        file_path = dest_path
        parameters = head.parameters if head.parameters else {}
        max_records = parameters.get("max_records", None)

        artifact_item = ArtifactItem(
            file_path=file_path,
            dag=self.artifact_dag,
            max_records=max_records,
            node_id=head.node_id if head else None,
        )
        if head.inputs is None:
            head.inputs = []
        head.inputs.append(artifact_item)
        
    def add_parameter(self, parameter):
        # print(f"Adding parameter {parameter} to head node {self.head.node_id if self.head else None}")

        # print(f"{len(parameter)} type: {type(parameter)}, existing parameters: {self.get_head().parameters if self.head else None}")

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

        # print(f"Saving DAG to yaml. artifact_dag: {self.artifact_dag}")

        if file_path is None:
            file_path = os.path.join(self.dag_dir, "dag.yaml")
        with open(file_path, "w") as file:

            # for node in self.nodes.values():
                # node.diff = Artifact.prune_empty(node.diff)

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
        dot = Digraph()

        dot.attr(
            # rankdir="TB",
            rankdir="LR",
            bgcolor="transparent",
            pad="0.15",
            nodesep="0.7",
            ranksep="1.0",
            splines="polyline",
        )

        dot.attr(
            "node",
            shape="plain",
            fontname="Helvetica",
            margin="0.12",
            width="3.2",
            height="1.2",
            fixedsize="false",
        )

        dot.attr(
            "edge",
            color="#64748B",
            penwidth="3.0",
            arrowsize="1.3",
            arrowhead="normal",
        )

        for node_id, node in self.nodes.items():
            dot.node(node_id, f"<{node.node_label()}>")

        for node_id, node in self.nodes.items():
            for parent_id in node.parents:
                spacer = f"{parent_id}_{node_id}_spacer"
                dot.node(spacer, label="", shape="point", width="0.01", height="0.01", style="invis")
                dot.edge(
                    parent_id,
                    spacer,
                    arrowhead="normal",
                    # minlen="0.1",
                )
                dot.edge(
                    spacer, 
                    node_id,
                    arrowhead="none", 
                    dir="none",
                    # constraint="false",
                    style="invis",
                )

        output_path = os.path.join(self.dag_dir, "dag")
        dot.render(output_path, format="svg", cleanup=True, view=view)
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
            # # print(f"{parent_node.node_id} outputs: {len(parent_node.outputs)}")

            for output in parent_node.outputs:
                if output not in node.inputs:
                    node.inputs.append(output)

        # before = Artifact.snapshot(node.inputs, node.outputs)

        print(f"Running node {node_id} {node.label} ({node.node_type}).")
        # print(repr(node))
        node.run()

        # for artifact in node.inputs + node.outputs:
            # artifact.refresh()
        # after = Artifact.snapshot(node.inputs, node.outputs)
        # changes = Artifact.diff_snapshots(before, after)
        # node.diff = changes
        # # print()
        # # print(f"Changes after running node {node_id}:")
        # p# print(changes)

        source_code = inspect.getsource(node.run)
        source_code_path = os.path.join(node.node_dir, f"{node.node_id}-script.py")
        with open(source_code_path, "w") as f:
            f.write(source_code)

        # # print(f"Finished running node {node_id}. parameters: {node.parameters}, inputs: {[i.file_path for i in node.inputs]}, outputs: {[o.file_path for o in node.outputs]}")
        if node.parameters and "last_run_source" in node.parameters:
            self.nodes[node_id].parameters["last_run_source"] = source_code_path
        
        if node_id in self.children:
            for child_id in self.children[node_id]:
                child_node = self.nodes[child_id]
                if not child_node.visited:
                    self._run_node(child_id)

    def run(self):
        if self.head is None:
            # # print("No head node set, cannot run DAG.")
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
                 label="Import Data",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
                 inputs=[],
                 outputs=[],
                 origin=False,
                 ):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            origin=origin,
            description="Imports external astronomical catalogs into the pipeline.",
        )


    def node_configure(self):
        if not self.parameters or "script" not in self.parameters:
            # write template script to node directory
            template_script = """# Example script for NodeImport.

# # print(f"*** Running in NodeImport...")
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
            label=d["label"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        """ Import artifact. """
        if len(self.inputs) > 0:
            artifact = self.inputs[0]
            
            # print(f"Running NodeImport with artifact {artifact.file_path} and parameters {self.parameters}")

            active_columns = {}

            script = self.parameters.get("script", "")

            with open(script, "r") as f:
                code = f.read()
                exec(code, {"parameters": self.parameters, "inputs": self.inputs, "outputs": self.outputs, "active_columns": active_columns})
            
            # print(f"Active columns after running script: {active_columns}")

            if self.parameters is not None and "max_records" in self.parameters:
                max_records = self.parameters["max_records"]
            else:
                max_records = 10

            if active_columns:
                # print(f"Selected columns from script: {active_columns}")
                artifact.set_active_columns(active_columns)
            
            artifact.dag = self.artifact_dag
            node_dir = os.path.join(self.dag_dir, self.node_id)
            artifact.file_path = os.path.join(node_dir, os.path.basename(artifact.file_path))
            artifact.max_records = max_records
            artifact._load_from_file()

            # print(f"Importing artifact {artifact.file_path} with dag {self.artifact_dag} and node_id {self.node_id}")

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
                 label="Export Data",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            description="Writes pipeline artifacts to persistent storage.",
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
                # print(f"Exporting artifact {artifact.file_path} with dag {self.artifact_dag} and node_id {self.node_id}")
                artifact.dag = self.artifact_dag
                
                # artifact._load_from_file()
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
                 label="Generic Node",
                 node_id=None,
                 parents=[],
                 parameters: dict[str, Any] | None = None,
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            description="Passes artifacts through the pipeline unchanged.",
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
            artifact = self.inputs.pop()
            data = Table.read(artifact.file_path)
            
            df = data.to_pandas()

            self.outputs = [artifact]



class NodeEDA(Node):
    def __init__(
            self,
            dag_dir=None,
            node_type="eda",
            label="Exploratory Data Analysis",
            node_id=None,
            parents=[],
            parameters={},
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label=label,
            description="Generates exploratory analysis and summary visualizations.",
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

        if len(self.inputs) == 0:
            # # print("No input artifact for EDA node.")
            return

        artifact = self.inputs[0]
        print(f"====Running EDA on artifact {artifact.file_path}. ")

        if self.parameters is not None and "title" in self.parameters:
            title = self.parameters["title"]
        else:
            title = "Exploratory Data Analysis"

        table = Table()
        artifact_dag = artifact.dag

        for col in artifact.active_columns:
            if col not in artifact.columns:
                # print(f"Column {col} not found in artifact columns {artifact.columns}. Skipping.")
                continue
            col_data = artifact.columns[col].latest_at(target_node_id=self.node_id, dag=artifact_dag)
            if col_data is not None:
                table[col] = col_data

        columns = artifact.active_columns

        dataset_eda(table=table, 
                    columns=columns, 
                    save_dir=self.node_dir, 
                    title=title,
                    )
        
        self.outputs = [artifact]


class NodeScript(Node):
    """
    A node that runs a user-provided script for testing the pipeline's ability to run arbitrary code.

    """

    def __init__(self,
                 dag_dir=None,
                 node_type="script",
                 label="Script Node",
                 node_id=None,
                 parents=[],
                 parameters={"script": None},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            description="Executes a user-defined data processing script.",
        )
    
    def node_configure(self):
        if self.parameters['script'] is None:
            # write template script to node directory
            template_script = """# Example script for NodeScript. This script will run by default. You can run your own script by setting the 'script' parameter in this node to the path of the script you want to run. Use this as a template and save the script in your catalog directory.

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

        print(f"Running NodeScript {self.label}")

        if len(self.inputs) > 0:
            artifact = self.inputs[0]

            table = artifact.to_table(self.node_id)
            df = table.to_pandas()
            columns = artifact.active_columns if artifact.active_columns else table.colnames

            script = self.parameters.get("script", "")

            namespace = {
                    "df": df,
                    "parameters": self.parameters,
                    "inputs": self.inputs,
                    "outputs": self.outputs,
                    "columns": columns,
                    }

            try:
                with open(script, "r") as f:
                    code = f.read()
                    exec(code, namespace)

            except Exception as e:
                # print(f"NodeScript failed to execute script {script} with error: {e}")
                raise e

            df = namespace["df"]
            columns = namespace["columns"]

            for col in df.columns:
                if col not in artifact.active_columns:
                    artifact.active_columns[col] = {}
                artifact.add_column_version(col, self.node_id, df[col])

            artifact.set_active_columns(columns)

            # # print(f"Executed script {script} on data with {len(df)} rows and {len(df.columns)} columns.")
            # # print(df)

            self.outputs = [artifact] 



class NodeEDAScript(Node):
    """
    A node that runs EDA (color-color, histogram, or sky_dist) using a  user-provided script to specify parameters and columns.

    """

    def __init__(self,
                 dag_dir=None,
                 node_type="eda-script",
                 label="EDA Script Node",
                 node_id=None,
                 parents=[],
                 parameters={"script": None, "eda_type": "histogram"},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            description="Runs EDA using user-defined parameters and column selection script.",
        )
    
    def node_configure(self):
        if "script" not in self.parameters:
            # write template script to node directory
            template_script = """# Example script for NodeEDAScript. This script will run by default. You can run your own script by setting the 'script' parameter in this node to the path of the script you want to run. Use this as a template and save the script in your catalog directory.

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
        d["type"] = "NodeEDAScript"
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

        print(f"Running NodeEDAScript {self.label}")

        if len(self.inputs) > 0:
            artifact = self.inputs[0]

            table = artifact.to_table(self.node_id)
            df = table.to_pandas()
            columns = artifact.active_columns if artifact.active_columns else table.colnames
            columns_original = columns.copy()

            script = self.parameters.get("script", "")
            eda_type = self.parameters.get("eda_type", "histogram")

            pair_plots = []

            namespace = {
                    "df": df,
                    "parameters": self.parameters,
                    "inputs": self.inputs,
                    "outputs": self.outputs,
                    "columns": columns,
                    "pair_plots": pair_plots,
                    }

            try:
                with open(script, "r") as f:
                    code = f.read()
                    exec(code, namespace)

            except Exception as e:
                # print(f"NodeScript failed to execute script {script} with error: {e}")
                raise e

            df = namespace["df"]
            columns = namespace["columns"]
            pair_plots = namespace["pair_plots"]

            # for col in df.columns:
                # if col not in artifact.active_columns:
                    # artifact.active_columns[col] = {}
                # artifact.add_column_version(col, self.node_id, df[col])

            # # print(f"Executed script {script} on data with {len(df)} rows and {len(df.columns)} columns.")
            # # print(df)

            if self.parameters is not None and "title" in self.parameters:
                title = self.parameters["title"]
            else:
                title = "Exploratory Data Analysis"

            table = Table()
            artifact_dag = artifact.dag

            for col in artifact.active_columns:
                if col not in columns:
                    # print(f"Column {col} not found in artifact columns {artifact.columns}. Skipping.")
                    continue
                col_data = artifact.columns[col].latest_at(target_node_id=self.node_id, dag=artifact_dag)
                if col_data is not None:
                    table[col] = col_data

            # columns = artifact.active_columns


            if eda_type == "histogram":
                eda_histogram(table=table, 
                            columns=columns, 
                            save_dir=self.node_dir, 
                            title=title,
                            )
            elif eda_type == "color-color":
                eda_color_color(table=table, 
                                columns=columns, 
                                save_dir=self.node_dir, 
                                title=title,
                                )
            elif eda_type == "sky-distribution":
                eda_sky_distribution(table=table, 
                            columns=columns, 
                            save_dir=self.node_dir, 
                            title=title,
                            )
            elif eda_type == "pair-plot":
                eda_pairplot(table=table, 
                            columns=columns, 
                            pair_plots=pair_plots,
                            save_dir=self.node_dir, 
                            title=title,
                            )
            else:
                # print(f"Unknown eda_type {eda_type}. Supported types are 'histogram', 'color-color', and 'sky_dist'.")
                raise ValueError(f"Unknown eda_type {eda_type}. Supported types are 'histogram', 'color-color', and 'sky-distribution'.")
            artifact.active_columns = columns_original
            self.outputs = [artifact]







class NodePhotometricDataset(Node):
    """
    A node that constructs a dataset.

    """

    def __init__(self,
                 dag_dir=None,
                 node_type="photometric-dataset",
                 label="Photometric Dataset Node",
                 node_id=None,
                 parents=[],
                 parameters={"script": None},
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            label=label,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            description="Constructs a photmetric dataset.",
        )
    
    def node_configure(self):
        if "script" not in self.parameters:
            # write template script to node directory
            template_script = """# Example script 

"""         
            script_path = os.path.join(self.node_dir, f"script.py")

            os.makedirs(self.node_dir, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(template_script)
            self.parameters["script"] = script_path
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodePhotometricDataset"
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

        print(f"Running NodePhotometricDataset {self.label} parameters: {self.parameters}")

        if len(self.inputs) > 0:
            artifact = self.inputs[0]

            table = artifact.to_table(self.node_id)
            df = table.to_pandas()

            columns = artifact.active_columns if artifact.active_columns else table.colnames
            columns_original = columns.copy()

            script = self.parameters.get("script", "")

            namespace = {
                    "df": df,
                    "parameters": self.parameters,
                    "inputs": self.inputs,
                    "outputs": self.outputs,
                    "columns": columns,
                    }

            try:
                with open(script, "r") as f:
                    code = f.read()
                    exec(code, namespace)

            except Exception as e:
                # print(f"NodeScript failed to execute script {script} with error: {e}")
                raise e

            df = namespace["df"]
            columns = namespace["columns"]

            table = Table()
            artifact_dag = artifact.dag

            for col in artifact.active_columns:
                if col not in columns:
                    # print(f"Column {col} not found in artifact columns {artifact.columns}. Skipping.")
                    continue
                col_data = artifact.columns[col].latest_at(target_node_id=self.node_id, dag=artifact_dag)
                if col_data is not None:
                    table[col] = col_data


            dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(self.parameters.get("dataset"))
            dataset.feature_names = columns

            for row in tqdm(df.itertuples(), total=len(df), desc="Building Photometric Dataset"):

                target_ra = row.ra
                target_dec = row.dec
                photometric_features = np.zeros((6, 3), dtype=np.float32)
                for bi, band in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
                    photometric_features[bi] = [
                        getattr(row, f"{band}_psfFlux_arcsinh", 0.0),
                        # getattr(row, f"{band}_psfFluxErr_arcsinh", 0.0),
                        getattr(row, f"{band}_psfFlux_SNR_log", 0.0),
                        getattr(row, f"{band}_psfFlux_mag", 0.0),
                        # getattr(row, f"{band}_psfFlux_bad_flag", 0.0),
                    ]
                photometric_features = np.hstack([photometric_features.flatten(),
                    getattr(row, "color_ug", np.nan),
                    getattr(row, "color_ur", np.nan),
                    getattr(row, "color_ui", np.nan),
                    getattr(row, "color_uz", np.nan),
                    getattr(row, "color_uy", np.nan),
                    getattr(row, "color_gr", np.nan),
                    getattr(row, "color_gi", np.nan),
                    getattr(row, "color_gz", np.nan),
                    getattr(row, "color_gy", np.nan),
                    getattr(row, "color_ri", np.nan),
                    getattr(row, "color_rz", np.nan),
                    getattr(row, "color_ry", np.nan),
                    getattr(row, "color_iz", np.nan),
                    getattr(row, "color_iy", np.nan),
                    getattr(row, "color_zy", np.nan),
                    getattr(row, "curvature_ug_gr", np.nan),
                    getattr(row, "curvature_gr_ri", np.nan),
                    getattr(row, "curvature_ri_iz", np.nan),
                    getattr(row, "curvature_iz_zy", np.nan),])

                hdu_phot = fits.ImageHDU(data=photometric_features, name="PHOTO")
                hdu_phot.header['label'] = int(row.label) if hasattr(row, "label") else 0
                hdu_phot.header['ra'] = float(target_ra)
                hdu_phot.header['dec'] = float(target_dec)
                hdu_phot.header['objectId'] = int(row.objectId)

                if (dataset.contains(row.objectId)):
                    dataset.update(row.objectId, hdu_phot)
                else:
                    dataset.append(hdu_phot)



            # columns = artifact.active_columns


            artifact.active_columns = columns_original
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

            # data1 = Table.read(artifact1.file_path)
            # data2 = Table.read(artifact2.file_path)
            data1 = artifact1.to_table(self.node_id)
            data2 = artifact2.to_table(self.node_id)
            df1 = data1.to_pandas()
            df2 = data2.to_pandas()

            print("df1", df1)
            print("df2", df2)

            # return

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

            margin_deg = 0.2
            df2_near = df2[
                (df2["ra"] >= df1["ra"].min() - margin_deg) &
                (df2["ra"] <= df1["ra"].max() + margin_deg) &
                (df2["dec"] >= df1["dec"].min() - margin_deg) &
                (df2["dec"] <= df1["dec"].max() + margin_deg)
            ].copy()

            print("df2_near rows:", len(df2_near))
            print(df2_near[["ra", "dec"]].head())


            c1 = SkyCoord(df1["ra"].to_numpy() * u.deg,
                            df1["dec"].to_numpy() * u.deg)
            c2 = SkyCoord(df2["ra"].to_numpy() * u.deg,
                            df2["dec"].to_numpy() * u.deg)
            idx, sep2d, _ = c1.match_to_catalog_sky(c2)
            sep_arcsec = sep2d.to(u.arcsec).value

            m = sep_arcsec < self.parameters.get("max_sep_arcsec", 1)

            print("min sep arcsec:", sep_arcsec.min())
            print("p01 sep arcsec:", np.percentile(sep_arcsec, 1))
            print("p05 sep arcsec:", np.percentile(sep_arcsec, 5))
            print("median sep arcsec:", np.median(sep_arcsec))

            for r in [1, 2, 5, 10, 30, 60]:
                print(f"matches within {r} arcsec:", np.sum(sep_arcsec < r))

            df1_matched = df1[m].reset_index(drop=True)
            df2_matched = df2.iloc[idx[m]].reset_index(drop=True)

            duplicate_cols = set(df1_matched.columns) & set(df2_matched.columns)

            df2_matched = df2_matched.rename(
                columns={col: f"{col}_2" for col in duplicate_cols}
            )

            df_matched = pd.concat([df1_matched, df2_matched], axis=1)

            df_matched["matched_idx"] = idx[m]
            df_matched["sep_arcsec"] = sep_arcsec[m]

            # metadata columns
            df_matched["matched_idx"] = idx[m]
            df_matched["sep_arcsec"] = sep_arcsec[m]

            df1_columns = artifact1.active_columns if artifact1.active_columns else {}
            df2_columns = artifact2.active_columns if artifact2.active_columns else {}

            matched_columns = {}
            for col in df1_columns:
                matched_columns[col] = df1_columns[col]
            for col in df2_columns:
                if col not in matched_columns:
                    matched_columns[col] = df2_columns[col]

            output_artifact = ArtifactItem(
                file_path=os.path.join(self.dag_dir, self.node_id, "joined.fits"),
                dag=self.artifact_dag,
                node_id=self.node_id,
                active_columns=matched_columns,
            )

            print("df_matched", df_matched)

            table = Table.from_pandas(df_matched)
            print(f"** JOIN Number in df_matched: {len(table)}")

            output_artifact.load_from_table(table, matched_columns)

            self.outputs = [output_artifact]
