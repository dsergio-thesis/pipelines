
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Dict, Literal, Optional, Tuple
import os
import argparse
import astropy.units as u
from astropy.coordinates import SkyCoord
import importlib
import csv
from pathlib import Path
from typing import Dict

from astroos_pipelines.utils.formatting import *
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])

CoordFormat = Literal["hmsdms", "deg"]

@dataclass
class CoordSpec:
    """ Specification for a sky region coordinate. """
    key: str
    value: str
    sky_coord: Optional[SkyCoord] = field(default=None, compare=False)
    fmt: CoordFormat = "hmsdms"
    radius_arcmin: float = 5.0
    selected: bool = False

    def __repr__(self):
        rows = [(self.key, self.value, self.fmt, self.radius_arcmin, self.selected)]
        return coord_table(rows, title="Target Coordinate Specification (CoordSpec)") 


class AstroosConfig:
    """ Configuration for the astronomy pipeline. Loads from environment variables and CLI args. """

    def __init__(
            self,
            dataset_dir: Path,
            pipeline_dir: Path,
            sky_regions_csv: Path,
            max_records: int,
            dataset_name: str = None,
            pipeline_name: str = None,
            option_create: bool = False,
            node_type: str = "generic", # NodeGeneric default
            node_label: str = None,
            input_artifact: str = None,
            parameter: (str, str) = None,
            sky_region_target_selected: str = None,
            sky_region_target_radius_arcmin: float = None,
            labels_def_file: Path = None,
            ):
        self.dataset_dir = dataset_dir
        self.pipeline_dir = pipeline_dir
        self.sky_regions_csv = sky_regions_csv
        self.dataset_name = dataset_name
        self.pipeline_name = pipeline_name
        self.option_create = option_create
        self.node_type = node_type
        self.node_label = node_label
        self.input_artifact = input_artifact
        self.parameter = parameter

        self.max_records = max_records

        self.sky_region_targets = self.load_coords_from_csv()
        self.sky_region_target_selected = sky_region_target_selected

        self.labels_def_file = labels_def_file
    
    @staticmethod
    def clean_path(value: str) -> Path:
        return Path(value.strip().strip('"').strip("'"))

    @staticmethod
    def clean_str(value: str) -> str:
        return value.strip().strip('"').strip("'")
    
    @classmethod
    def from_env(cls):
        """ Load configuration from environment variables. """
        def env(key: str, default: Optional[str] = None) -> str:
            v = os.getenv(key, default)
            if v is None:
                raise RuntimeError(f"Missing required env var: {key}")
            return v

        return cls(
            dataset_dir=cls.clean_path(env("DATASET_DIR")).expanduser(),
            pipeline_dir=Path(env("PIPELINE_DIR")).expanduser(),
            max_records=int(env("MAX_RECORDS", "6")),
            sky_regions_csv=cls.clean_path(env("SKY_REGIONS_CSV")).expanduser(),
            labels_def_file=cls.clean_path(env("LABELS_DEF_FILE")).expanduser(),
        )

    
    @classmethod
    def from_cli(cls, argv: Optional[list[str]] = None):
        """ Load configuration from CLI args. """
        parser = cls.build_arg_parser()
        args = parser.parse_args(argv)

        # start from env
        base = cls.from_env()

        # merge CLI and env
        cfg = cls(
            dataset_dir=base.dataset_dir,
            pipeline_dir=base.pipeline_dir,
            dataset_name=args.dataset_name,
            pipeline_name=args.pipeline_name,
            option_create=args.create,
            input_artifact=args.input_artifact,
            parameter=tuple(args.parameter) if args.parameter else None,
            node_type=args.node_type,
            node_label=args.node_label,
            sky_regions_csv=base.sky_regions_csv,
            sky_region_target_selected=args.target,
            max_records=args.max_records if args.max_records is not None else base.max_records,
        )

        # optional runtime coord override
        if args.target:
            if args.target not in cfg.sky_region_targets:
                raise ValueError(
                    f"Unknown target '{args.target}'. "
                    f"Valid targets: {list(cfg.sky_region_targets)}"
                )

        return cfg


    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Run the astronomy pipeline",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        p.add_argument("-d", "--dataset-name", type=str, help="Dataset name")
        p.add_argument("-n", "--pipeline-name", type=str, help="Pipeline name")
        p.add_argument("-m", "--max-records", type=int, help="Max records to fetch from query")
        p.add_argument("-c", "--create", action="store_true", help="Create new  node")
        p.add_argument("-i", "--input-artifact", type=str, help="Path to input artifact")
        p.add_argument("-t", "--node-type", type=str, default="generic", help="Node type for new node")
        p.add_argument("-l", "--node-label", type=str, help="Label for new node (defaults to node type)")
        p.add_argument("--target", type=str, help="Target key to override coordinates")
        p.add_argument("-p", "--parameter", nargs=2, metavar=("KEY", "VALUE"), help="Additional parameter as key value pair")
        return p

    def __repr__(self):

        rows = [
            ("dataset_dir", "", "DATASET_DIR", self.dataset_dir),
            ("pipeline_dir", "", "PIPELINE_DIR", self.pipeline_dir),
            ("dataset_name", "-d / --dataset-name", "", self.dataset_name),
            ("pipeline_name", "-n / --pipeline-name", "", self.pipeline_name),
            ("max_records", "-m / --max-records", "MAX_RECORDS", self.max_records),
            ("option_create", "-c / --create", "", self.option_create),
            ("node_type", "-t / --node-type", "", self.node_type),
            ("node_label", "-l / --node-label", "", self.node_label),
            ("input_artifact", "-i / --input-artifact", "", self.input_artifact),
            ("parameter", "-p / --parameter", "", self.parameter),
            ("sky_regions_csv", "", "SKY_REGION_LABELS", self.sky_regions_csv),
            ("sky_region_target_selected", "--target", "", self.sky_region_target_selected),        
            ]
        config_str = ascii_config_table(rows, title="Pipeline Configuration")

        if self.sky_region_targets:
            specs = []
            for _, spec in self.sky_region_targets.items():
                specs.append(spec)
            config_str += "\n\n" + coord_table(
                [(s.key, s.value, s.fmt, s.radius_arcmin, s.selected) for s in specs],
                title="Sky Region Targets (CoordSpec)"
            )
        return config_str
    
    def set_target(self, key: str) -> Tuple[SkyCoord, u.Quantity]:
        key = key.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if key not in self.sky_region_targets:
            print()
            raise RuntimeError(f"{key} is not in {self.sky_region_targets}")
        
        spec = self.sky_region_targets[key]
        spec.selected = True

        self.sky_region_target_selected = key

    from pathlib import Path

    def set_env_var(self, key: str, value: str):
        env_path = Path("env.sh")
        value = str(value).strip().strip('"').strip("'")  # clean value

        lines = []
        found = False

        if env_path.exists():
            lines = env_path.read_text().splitlines()

        new_lines = []

        for line in lines:
            stripped = line.strip()

            # preserve comments/blank lines
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            if stripped.startswith(f"export {key}="):
                new_lines.append(f'export {key}="{value}"')
                found = True
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f'export {key}="{value}"')

        env_path.write_text("\n".join(new_lines) + "\n")

    def load_coords_from_csv(self) -> Dict[str, CoordSpec]:
        coords = {}
        path = self.sky_regions_csv 

        # if file doesn't exist, return empty dict (allows for optional custom label files)
        if not Path(path).is_file():
            print(f"Warning: Label definition file '{path}' not found. No coordinates loaded.")
            return coords

        with open(path, newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                label = row["Label"].strip()
                ra_dec = row["RA_DEC"].strip()
                fmt = row["Format"].strip()
                r = float(row["Size_arcmin"])
                r = r * u.arcmin

                # normalize key
                key = label.lower().replace(" ", "_").replace("(", "").replace(")", "")

                if fmt not in ("hmsdms", "deg"):
                    raise ValueError(f"Invalid format '{fmt}' for label '{label}' in CSV. Must be 'hmsdms' or 'deg'.")
                
                if fmt == "hmsdms":
                    coord = SkyCoord(ra_dec, unit=(u.hourangle, u.deg))  # parse to validate
                else:
                    ra, dec = map(float, ra_dec.split())
                    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)  # parse to validate

                coords[key] = CoordSpec(key=key, value=ra_dec, sky_coord=coord, fmt=fmt, radius_arcmin=r)

        return coords
