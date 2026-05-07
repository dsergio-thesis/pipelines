from __future__ import annotations

from pathlib import Path

import os
from typing import Any

from astropy.table import Table
from pandas.util import hash_pandas_object
import pandas as pd

from dataclasses import dataclass
from pprint import pprint
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from pandas.util import hash_pandas_object


class ArtifactMergeConflict(Exception):
    pass


class ArtifactDAG:
    def __init__(self):
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)

    def add_edge(self, parent: str, child: str) -> None:
        print(f"Adding edge from '{parent}' to '{child}'")
        self.children[parent].add(child)
        self.parents[child].add(parent)

        # Ensure parent exists in parents map too
        self.parents[parent]

    def ancestors(self, node_id: str, include_self: bool = True) -> set[str]:
        seen = {node_id} if include_self else set()
        stack = list(self.parents[node_id])

        while stack:
            node = stack.pop()

            if node in seen:
                continue

            seen.add(node)
            stack.extend(self.parents[node])

        return seen

    def is_ancestor(self, maybe_ancestor: str, node_id: str) -> bool:
        return maybe_ancestor in self.ancestors(node_id, include_self=True)

    def comparable(self, a: str, b: str) -> bool:
        """
        True if a and b are on the same lineage.
        """
        return self.is_ancestor(a, b) or self.is_ancestor(b, a)

    def to_dict(self):
        ret = {
            "parents": {
                node: sorted(parents)
                for node, parents in self.parents.items()
                if parents
            }
        }
        print("DAG to_dict:", ret)
        return ret

    @classmethod
    def from_dict(cls, d):
        print("Creating DAG from dict:", d)
        dag = cls()

        for child, parents in d.get("parents", {}).items():
            for parent in parents:
                dag.add_edge(parent, child)

        return dag

    def __repr__(self):
        return f"ArtifactDAG(edges={self.to_dict()['parents']})"


@dataclass
class ColumnVersion:
    node_id: str
    data: pd.Series
    hash: int


class ArtifactCol:
    def __init__(self, name: str):
        self.name = name
        self.versions: list[ColumnVersion] = []

    def add_version(self, node_id: str, data) -> None:
        series = self._to_series(data)

        self.versions.append(
            ColumnVersion(
                node_id=node_id,
                data=series,
                hash=int(hash_pandas_object(series, index=True).sum()),
            )
        )

    def add_version_from_disk_csv(self, node_id: str, file_path: str) -> None:
        series = pd.read_csv(file_path).iloc[:, 0]

        self.versions.append(
            ColumnVersion(
                node_id=node_id,
                data=series,
                hash=int(hash_pandas_object(series, index=True).sum()),
            )
        )

    def add_version_from_disk(self, node_id: str, file_path: str) -> None:
        df = pd.read_parquet(file_path)
        series = df.iloc[:, 0]

        self.versions.append(
            ColumnVersion(
                node_id=node_id,
                data=series,
                hash=int(hash_pandas_object(series, index=True).sum()),
            )
        )

    def latest_at(self, target_node_id: str, dag: ArtifactDAG) -> pd.Series | None:
        valid_nodes = dag.ancestors(target_node_id, include_self=True)

        candidates = [
            version
            for version in self.versions
            if version.node_id in valid_nodes
        ]

        if not candidates:
            return None

        # Remove candidates that are older ancestors of another candidate.
        # Example: n1 and n3 both exist; n1 is ancestor of n3, so n3 supersedes n1.
        maximal = []

        for candidate in candidates:
            is_superseded = False

            for other in candidates:
                if candidate is other:
                    continue

                if dag.is_ancestor(candidate.node_id, other.node_id):
                    is_superseded = True
                    break

            if not is_superseded:
                maximal.append(candidate)

        if len(maximal) > 1:
            conflict_nodes = [v.node_id for v in maximal]

            raise ArtifactMergeConflict(
                f"Merge conflict for column '{self.name}' at node '{target_node_id}'. "
                f"Conflicting versions from nodes: {conflict_nodes}. "
                f"Add an explicit merged version of column '{self.name}' at '{target_node_id}' "
                f"or before it."
            )

        return maximal[0].data

    @staticmethod
    def _to_series(data) -> pd.Series:
        if isinstance(data, pd.Series):
            return data.reset_index(drop=True)

        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValueError("Column data DataFrame must have exactly one column")

            return data.iloc[:, 0].reset_index(drop=True)

        return pd.Series(data).reset_index(drop=True)

    def __repr__(self) -> str:
        lines = [f"ArtifactCol({self.name})"]

        for version in self.versions:
            lines.append(f"  {version.node_id}: hash={version.hash}")
            lines.append(str(version.data.head()))

        return "\n".join(lines)


class ArtifactItem:
    def __init__(self, file_path: str, dag: ArtifactDAG = None, node_id: str = None):
        self.file_path = file_path
        self.dag = dag 
        self.columns: dict[str, ArtifactCol] = {}
        self.node_id = node_id

        if os.path.exists(file_path) and node_id is not None:
            self._load_from_file()

    def add_column(self, col_name: str, node_id: str, data) -> None:
        if col_name not in self.columns:
            self.columns[col_name] = ArtifactCol(col_name)

        self.columns[col_name].add_version(node_id, data)

    def to_df(self, node_id: str) -> pd.DataFrame:
        if self.dag is None: 
            raise ValueError("DAG is required to convert to DataFrame")
        combined = {}

        for col_name, artifact_col in self.columns.items():
            series = artifact_col.latest_at(node_id, self.dag)

            if series is not None:
                combined[col_name] = series

        return pd.DataFrame(combined)

    def to_csv(self, node_id: str) -> None:
        df = self.to_df(node_id)
        df.to_csv(self.file_path, index=False)

    def materialize(self, node_id: str) -> None:
        df = self.to_df(node_id)

        ext = Path(self.file_path).suffix.lower()

        if ext == ".csv":
            df.to_csv(self.file_path, index=False)

        elif ext == ".parquet":
            df.to_parquet(self.file_path, index=False)

        elif ext in {".fits", ".fit"}:
            table = Table.from_pandas(df)
            table.write(self.file_path, overwrite=True)

        else:
            raise ValueError(f"Unsupported output format: {ext}")

    def _load_from_file(self) -> None:
        try:
            table = Table.read(self.file_path)
            df = table.to_pandas()

            for col in df.columns:
                self.add_column(col, node_id=self.node_id, data=df[col])

        except Exception as e:
            print(f"Error reading file {self.file_path}: {e}")
            self.columns = {}

    def to_dict(self) -> dict:

        # get dir from file_path and ensure it exists
        if not self.file_path:
            raise ValueError("file_path is required to save column versions to disk")
        column_store_dir = os.path.dirname(self.file_path)
        store_dir = Path(column_store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        columns_dict = {}

        for col_name, artifact_col in self.columns.items():
            columns_dict[col_name] = []

            for version in artifact_col.versions:
                safe_col = str(col_name).replace("/", "_")
                safe_node = str(version.node_id).replace("/", "_")

                version_file = (
                        store_dir 
                        / f"{Path(self.file_path).stem}__{safe_col}__{safe_node}.parquet"
                )

                if version.data is not None:
                    df = pd.DataFrame({col_name: version.data})
                    df.to_parquet(version_file, index=False)

                columns_dict[col_name].append({
                    "node_id": version.node_id,
                    "hash": version.hash,
                    "file_path": str(version_file),
                })

        return {
            "file_path": self.file_path,
            "node_id": self.node_id,
            "columns": columns_dict,
        }

    def to_dict_csv(self) -> dict:

        # get dir from file_path and ensure it exists
        if not self.file_path:
            raise ValueError("file_path is required to save column versions to disk")
        column_store_dir = os.path.dirname(self.file_path)

        store_dir = Path(column_store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        columns_dict = {}

        for col_name, artifact_col in self.columns.items():
            columns_dict[col_name] = []

            for version in artifact_col.versions:
                safe_col = str(col_name).replace("/", "_")
                safe_node = str(version.node_id).replace("/", "_")

                version_file = store_dir / f"{Path(self.file_path).stem}__{safe_col}__{safe_node}.csv"

                if version.data is not None:
                    pd.DataFrame({col_name: version.data}).to_csv(version_file, index=False)

                columns_dict[col_name].append(
                    {
                        "node_id": version.node_id,
                        "hash": version.hash,
                        "file_path": str(version_file),
                    }
                )

        return {
            "file_path": self.file_path,
            "node_id": self.node_id,
            "columns": columns_dict,
        }

    @classmethod
    def from_dict(cls, d: dict, dag: ArtifactDAG = None) -> "ArtifactItem":
        item = cls(
            file_path=d["file_path"],
            dag=dag,
            node_id=d.get("node_id"),
        )

        item.columns = {}

        for col_name, versions in d.get("columns", {}).items():
            artifact_col = ArtifactCol(col_name)

            for version in versions:
                artifact_col.add_version_from_disk(
                    node_id=version["node_id"],
                    file_path=version["file_path"],
                )

            item.columns[col_name] = artifact_col

        return item

    def __repr__(self) -> str:
        lines = [f"ArtifactItem(file_path={self.file_path}, node_id={self.node_id})"]

        c = 0
        for col_name, artifact_col in self.columns.items():
            lines.append(f"  Column: {col_name}")

            for version in artifact_col.versions:
                lines.append(f"    Version from node '{version.node_id}': hash={version.hash}")
                lines.append(str(version.data.head()))
            c += 1
            if c >= 3:
                lines.append("    ...")
                break

        return "\n".join(lines)




class Artifact:
    """DAG artifact, the data that flows between nodes in the DAG."""

    SUPPORTED_EXTENSIONS = {
        ".csv": "csv",
        ".fits": "fits",
        ".fit": "fits",
    }

    def __init__(
        self,
        name: str,
        file_path: str,
        columns: dict[str, Any] | None = None,
    ):
        self.name = name
        self.file_path = file_path
        self.file_type = self._infer_file_type(file_path)

        if columns is None:
            self.columns = {}
            self.set_columns()
        else:
            self.columns = columns

    @classmethod
    def _infer_file_type(cls, file_path: str) -> str:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type {file_ext}. Only csv and fits are supported."
            )

        return cls.SUPPORTED_EXTENSIONS[file_ext]

    def set_columns(self) -> None:
        """Read the artifact file and store column metadata + hashes."""
        self.columns = {}

        if not self.file_path or not os.path.exists(self.file_path):
            return

        try:
            table = Table.read(self.file_path)
            df = table.to_pandas()

            column_hashes = []

            for col in df.columns:
                col_hash = self._hash_column(df[col])
                column_hashes.append(col_hash)

                self.columns[col] = {
                    "type": str(df[col].dtype),
                    "hash": col_hash,
                }

            self.columns["_file"] = {
                "type": self.file_type,
                "hash": self._hash_values(column_hashes),
            }

        except Exception as e:
            print(f"Error reading file {self.file_path}: {e}")
            self.columns = {}

    @staticmethod
    def _hash_column(column) -> int:
        return int(hash_pandas_object(column, index=True).sum())

    @staticmethod
    def _hash_values(values: list[int]) -> int:
        return hash(tuple(values))

    def refresh(self) -> None:
        """Re-read the file and update stored hashes."""
        self.set_columns()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "columns": self.columns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Artifact":
        return cls(
            name=d["name"],
            file_path=d["file_path"],
            columns=d.get("columns"),
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Artifact):
            return False

        return (
            self.file_path == other.file_path
            )

    def __hash__(self) -> int:
        return hash((self.file_path))
    
        # file_hash = None

        # if self.columns:
            # file_hash = self.columns.get("_file", {}).get("hash")

        # return hash((self.name, self.file_path, file_hash))

    @staticmethod
    def snapshot(inputs: list["Artifact"], outputs: list["Artifact"]) -> dict:
        return {
            "inputs": {
                artifact.name: artifact.to_dict()
                for artifact in inputs
            },
            "outputs": {
                artifact.name: artifact.to_dict()
                for artifact in outputs
            },
        }

    @staticmethod
    def diff_snapshots(before: dict, after: dict) -> dict:
        return {
            "inputs": Artifact.diff_artifact_group(
                before.get("inputs", {}),
                after.get("inputs", {}),
            ),
            "outputs": Artifact.diff_artifact_group(
                before.get("outputs", {}),
                after.get("outputs", {}),
            ),
        }

    @staticmethod
    def diff_artifact_group(before_group: dict, after_group: dict) -> dict:
        before_names = set(before_group)
        after_names = set(after_group)

        added_artifacts = after_names - before_names
        removed_artifacts = before_names - after_names
        common_artifacts = before_names & after_names

        changed_artifacts = {}

        for name in common_artifacts:
            before_cols = before_group[name].get("columns", {})
            after_cols = after_group[name].get("columns", {})

            col_diff = Artifact.diff_columns(before_cols, after_cols)

            if col_diff:
                changed_artifacts[name] = col_diff

        return Artifact.prune_empty({
            "added_artifacts": added_artifacts,
            "removed_artifacts": removed_artifacts,
            "changed_artifacts": changed_artifacts,
        })

    @staticmethod
    def diff_columns(before_cols: dict | None, after_cols: dict | None) -> dict:
        before_cols = before_cols or {}
        after_cols = after_cols or {}

        before_names = set(before_cols)
        after_names = set(after_cols)

        added = after_names - before_names
        removed = before_names - after_names
        common = before_names & after_names

        changed = {
            col
            for col in common
            if before_cols[col] != after_cols[col]
        }

        return Artifact.prune_empty({
            "added": added,
            "removed": removed,
            "changed": changed,
        })

    @staticmethod
    def prune_empty(obj):
        if isinstance(obj, dict):
            cleaned = {
                k: Artifact.prune_empty(v)
                for k, v in obj.items()
            }

            return {
                k: v
                for k, v in cleaned.items()
                if v not in ({}, [], set(), None)
            }

        if isinstance(obj, list):
            return [Artifact.prune_empty(v) for v in obj]

        if isinstance(obj, set):
            return sorted(obj)

        return obj
