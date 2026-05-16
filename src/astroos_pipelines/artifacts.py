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
        # print(f"Adding edge from '{parent}' to '{child}'")
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
        # print("DAG to_dict:", ret)
        return ret

    @classmethod
    def from_dict(cls, d):
        # print("Creating DAG from dict:", d)
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
    data: pd.Series | None
    hash: int
    file_path: str | None = None

    def load_data(self, file_path: str, max_records: int = None, shuffle: bool = False) -> pd.Series:
        print(f"calling load_data for node_id={self.node_id}, max_records={max_records}, shuffle={shuffle}")

        self.file_path = self.file_path or file_path

        # only use cache for full unshuffled dataset
        if self.data is not None:
            return self.data[:max_records] if max_records is not None else self.data

        if self.file_path is None:
            raise ValueError(f"No data or file_path for version at node {self.node_id} {self.file_path}")
        
        # shuffle if requested
        if shuffle:
            df = pd.read_parquet(self.file_path)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            df = pd.read_parquet(self.file_path)

        # Only load up to max_records
        if max_records is not None:
            df = df.head(max_records)
        self.data = df.iloc[:, 0].reset_index(drop=True)

        return self.data


class ArtifactCol:
    def __init__(self, name: str, file_path: str = None):
        self.name = name
        self.versions: list[ColumnVersion] = []
        self.file_path = file_path

    def add_version(self, node_id: str, data) -> None:
        series = self._to_series(data)

        self.versions.append(
            ColumnVersion(
                node_id=node_id,
                data=series,
                hash=int(hash_pandas_object(series, index=True).sum()),
                file_path=self.file_path,
            )
        )

    def add_version_metadata(
        self,
        node_id: str,
        col_hash: int,
        file_path: str,
        ) -> None:
        self.versions.append(
            ColumnVersion(
                node_id=node_id,
                data=None,
                hash=col_hash,
                file_path=file_path,
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

    def latest_at(self, target_node_id: str, dag: ArtifactDAG, max_records: int = None) -> pd.Series | None:
        valid_nodes = dag.ancestors(target_node_id, include_self=True)

        candidates = [
            version
            for version in self.versions
            if version.node_id in valid_nodes
        ]

        if not candidates:
            return None

        maximal = []

        for candidate in candidates:
            is_superseded = False

            for other in candidates:
                if candidate is other:
                    continue

                # Do not let same-node versions supersede each other
                if candidate.node_id == other.node_id:
                    continue

                if dag.is_ancestor(candidate.node_id, other.node_id):
                    is_superseded = True
                    break

            if not is_superseded:
                maximal.append(candidate)

        if not maximal:
            raise ValueError(
                f"No maximal version found for column '{self.name}' at node '{target_node_id}'. "
                f"Candidates: {[v.node_id for v in candidates]}"
            )

        if len(maximal) > 1:
            conflict_nodes = [v.node_id for v in maximal]

            raise ArtifactMergeConflict(
                f"Merge conflict for column '{self.name}' at node '{target_node_id}'. "
                f"Conflicting versions from nodes: {conflict_nodes}."
            )

        return maximal[0].load_data(file_path=self.file_path, max_records=max_records)

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
            lines.append(
                f"  {version.node_id}: "
                f"hash={version.hash}, "
                f"file_path={version.file_path}, "
                f"loaded={version.data is not None}"
            )

            if version.data is not None:
                lines.append(str(version.data.head()))

        return "\n".join(lines)

class ArtifactItem:
    def __init__(self, 
                 file_path: str, 
                 dag: ArtifactDAG = None, 
                 node_id: str = None, 
                 columns: dict[str, ArtifactCol] = None,
                 active_columns: dict[str, dict] | None = None,
                 load_from_file: bool = True,
                 max_records: int = None,
                 ):
        self.file_path = file_path
        self.file_type = Path(file_path).suffix.lower()
        self.dag = dag 
        self.columns = columns or {}
        self.node_id = node_id
        self.max_records = max_records

        self.active_columns = active_columns 

        if os.path.exists(file_path) and node_id is not None and columns is None and load_from_file:
            self._load_from_file()
            for col in self.columns.values(): 
                col.file_path = file_path

    def set_active_columns(self, active_columns: dict[str, dict] | list[str] | None) -> None:
        if active_columns is None:
            self.active_columns = None
        elif isinstance(active_columns, list):
            self.active_columns = {col: {col} for col in active_columns}
        else:
            self.active_columns = active_columns

    def get_active_column_names(self) -> list[str]:
        if self.active_columns is None:
            return list(self.columns.keys())

        return [
            col for col in self.active_columns
            if col in self.columns
        ]

    def get_active_column_metadata(self, col_name: str) -> dict:
        if self.active_columns is None:
            return {}

        return self.active_columns.get(col_name, {})

    def add_column_version(self, col_name: str, node_id: str, data) -> None:
        if col_name not in self.columns:
            self.columns[col_name] = ArtifactCol(col_name, file_path=self.file_path)

        self.columns[col_name].add_version(node_id, data)

    def add_column_version_metadata(self, col_name: str, node_id: str, data) -> None:
        if col_name not in self.columns:
            self.columns[col_name] = ArtifactCol(col_name)

        self.columns[col_name].add_version_metadata(node_id, data)

    def get_latest_column_version(self, col_name: str, target_node_id: str) -> pd.Series | None:
        if self.dag is None:
            raise ValueError("DAG is required to get latest column version")

        if col_name not in self.columns:
            return None

        return self.columns[col_name].latest_at(target_node_id, self.dag)


    def to_df(self, node_id: str, col_names: list[str] = None) -> pd.DataFrame:
        if self.dag is None: 
            raise ValueError("DAG is required to convert to DataFrame")

        if col_names is not None:
            df_columns = col_names
        else:
            df_columns = self.get_active_column_names()

        combined = {}
        for col_name in df_columns:
            artifact_col = self.columns[col_name]
            series = artifact_col.latest_at(node_id, self.dag, max_records=self.max_records)
            # print(f"series: {series.head()}")

            if series is not None:
                combined[col_name] = series

        return pd.DataFrame(combined)

    def to_table(self, node_id: str, col_names: list[str] = None) -> Table:
        df = self.to_df(node_id, col_names)
        return Table.from_pandas(df)

    def materialize(self, node_id: str) -> None:
        first_col = next(iter(self.columns.values()), None)

        if (self.max_records is not None 
            and self.max_records < first_col.latest_at(node_id, self.dag).shape[0]
            ):
            print(f" --> max_records={self.max_records} is less than the number of records in the first column.")
            print(f"Increase max_records or set it to None to materialize all records for this node.")
            

        print(f"Materializing artifact to {self.file_path} at node {node_id}, max_records={self.max_records}...")
        df = self.to_df(node_id)

        ext = Path(self.file_path).suffix.lower()

        if ext == ".csv":
            df.to_csv(self.file_path, index=False)
        elif ext == ".parquet":
            df.to_parquet(self.file_path, index=False)
        elif ext in {".fits", ".fit"}:
            table = Table.from_pandas(df)
            table.write(self.file_path, format="fits", overwrite=True)

        else:
            raise ValueError(f"Unsupported output format: {ext}")

    def _load_from_file(self) -> None:
        try:
            ext = Path(self.file_path).suffix.lower()

            if ext not in {".csv", ".parquet", ".fits", ".fit"}:
                raise ValueError(f"Unsupported file format: {ext}")

            if ext == ".csv":
                df = pd.read_csv(self.file_path)
            elif ext == ".parquet":
                df = pd.read_parquet(self.file_path)
            elif ext in {".fits", ".fit"}:
                table = Table.read(self.file_path, format="fits", hdu=1)
                df = table.to_pandas()

            for col in df.columns:
                self.add_column_version(col, node_id=self.node_id, data=df[col])

        except Exception as e:
            print(f"Error reading file {self.file_path}: {e}")
            self.columns = {}
            raise e

    def load_from_table(self, table, columns):
        try:

            ext = Path(self.file_path).suffix.lower()

            if ext not in {".csv", ".parquet", ".fits", ".fit"}:
                raise ValueError(f"Unsupported file format: {ext}")

            df = table.to_pandas()
            
            
            active_columns = {}
            for col in df.columns:
                active_columns[col] = {}
            self.active_columns = columns

            for col in df.columns:
                self.add_column_version(col, node_id=self.node_id, data=df[col])

        except Exception as e:
            print(f"Error reading file {self.file_path}: {e}")
            self.columns = {}
            raise e


    def to_dict(self) -> dict:
        # get dir from file_path and ensure it exists
        if not self.file_path:
            raise ValueError("file_path is required to save column versions to disk")
        column_store_dir = os.path.dirname(self.file_path)
        store_dir = Path(column_store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)

        column_index_path = Path(column_store_dir) / f"{Path(self.file_path).stem}__column_index.parquet"
        active_columns_index_path = Path(column_store_dir) / f"{Path(self.file_path).stem}__active_columns.parquet"

        columns_dict = {}
        columns_rows = []

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
                columns_rows.append({
                    "column": col_name,
                    "node_id": version.node_id,
                    "hash": version.hash,
                    "data_path": str(version_file),
                })

        for col_name, artifact_col in self.columns.items():
            active_columns_dict[col_name] = metadata

            for version in artifact_col.versions:
                active_columns_rows.append({
                    "column": col_name,
                    "node_id": version.node_id,
                    "hash": version.hash,
                    "data_path": str(version_file),
                })

        pd.DataFrame(active_columns_rows).to_parquet(active_columns_index_path, index=False)
        pd.DataFrame(columns_rows).to_parquet(column_index_path, index=False)

        return {
            "file_path": self.file_path,
            "node_id": self.node_id,
            "column_index_path":  str(column_index_path),
            # "active_columns": self.active_columns,
            "active_columns_index_path": str(active_columns_index_path),
            "max_records": self.max_records,
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
            columns={},  # prevents _load_from_file()
            active_columns=d.get("active_columns"),
            active_columns_index_path=d.get("active_columns_index_path"),
            max_records=d.get("max_records"),
        )

        index_df = pd.read_parquet(d["column_index_path"])

        for _, row in index_df.iterrows():
            col_name = row["column"]

            if col_name not in item.columns:
                item.columns[col_name] = ArtifactCol(col_name)

            item.columns[col_name].add_version_metadata(
                node_id=row["node_id"],
                col_hash=int(row["hash"]),
                file_path=row["data_path"],
            )

        return item

    def __repr__(self) -> str:
        lines = [f"ArtifactItem(file_path={self.file_path}, node_id={self.node_id})"]

        c = 0
        for col_name, artifact_col in self.columns.items():
            lines.append(f"  Column: {col_name}")

            for version in artifact_col.versions:
                lines.append(f"    Version from node '{version.node_id}': hash={version.hash}")

                if version.data is not None:
                    lines.append(str(version.data.head()))
            c += 1
            if c >= 3:
                lines.append("    ...")
                break

        return "\n".join(lines)

