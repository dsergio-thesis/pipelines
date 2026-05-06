
from pandas.util import hash_pandas_object
import os
from astropy.table import Table


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
        
        # get file type, and only allow csv and fits
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".csv":
            self.file_type = "csv"
        elif file_ext in [".fits", ".fit"]:
            self.file_type = "fits"
        else:
            raise ValueError(f"Unsupported file type {file_ext} for artifact {name}. Only csv and fits are supported.")

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

    def snapshot(self):
        from pandas.util import hash_pandas_object
        if self.file_path is not None and os.path.exists(self.file_path):
            try:
                table = Table.read(self.file_path)
                df = table.to_pandas()
                return {
                    col: hash_pandas_object(df[col], index=True).sum() for col in df.columns
                }
            except Exception as e:
                # print(f"Error reading file {self.file_path} for snapshot: {e}")
                return None
        else:
            # print(f"File path {self.file_path} does not exist for snapshot.")
            return None

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


def snapshot(inputs: list[Artifact], outputs: list[Artifact]) -> dict:
    return {
        "inputs": {
            artifact.name: artifact.snapshot()
            for artifact in inputs
        },
        "outputs": {
            artifact.name: artifact.snapshot()
            for artifact in outputs
        },
    }


def diff_snapshots(before: dict, after: dict) -> dict:
    return {
        "inputs": diff_artifact_group(
            before.get("inputs", {}),
            after.get("inputs", {}),
        ),
        "outputs": diff_artifact_group(
            before.get("outputs", {}),
            after.get("outputs", {}),
        ),
    }


def diff_artifact_group(before_group: dict, after_group: dict) -> dict:
    before_names = set(before_group)
    after_names = set(after_group)

    added_artifacts = after_names - before_names
    removed_artifacts = before_names - after_names
    common_artifacts = before_names & after_names

    changed_artifacts = {}

    for name in common_artifacts:
        before_cols = before_group[name]
        after_cols = after_group[name]

        changed_artifacts[name] = diff_columns(before_cols, after_cols)

    return {
        "added_artifacts": added_artifacts,
        "removed_artifacts": removed_artifacts,
        "changed_artifacts": changed_artifacts,
    }


def diff_columns(before_cols: dict | None, after_cols: dict | None) -> dict:
    if before_cols is None and after_cols is None:
        return {
            "added": set(),
            "removed": set(),
            "changed": set(),
            "read_error": True,
        }

    if before_cols is None:
        return {
            "added": set(after_cols or {}),
            "removed": set(),
            "changed": set(),
            "read_error_before": True,
        }

    if after_cols is None:
        return {
            "added": set(),
            "removed": set(before_cols),
            "changed": set(),
            "read_error_after": True,
        }

    before_names = set(before_cols)
    after_names = set(after_cols)

    added = after_names - before_names
    removed = before_names - after_names
    common = before_names & after_names

    changed = {
        col for col in common
        if before_cols[col] != after_cols[col]
    }

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }

def prune_empty(obj):
    if isinstance(obj, dict):
        cleaned = {
            k: prune_empty(v)
            for k, v in obj.items()
        }

        return {
            k: v
            for k, v in cleaned.items()
            if v not in ({}, [], None)
        }

    if isinstance(obj, list):
        return [prune_empty(v) for v in obj]

    if isinstance(obj, set):
        return sorted(obj)

    return obj
