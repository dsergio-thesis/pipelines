
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

from utils.formatting_utils import ascii_kv_table
importlib.reload(sys.modules['utils.formatting_utils'])



@dataclass(frozen=True)
class AstroosConfig:
    dataset_dir: Path
    dataset_name: str
    pipeline_dir: Path
    pipeline_name: str
    pipeline_minor_version: int
    label_def_file: str


    # default targets in SkyCoord:
    coords: Dict[str, SkyCoord] = field(default_factory=lambda: {
        "virgo_cluster": SkyCoord("12 30 49.423 +12 23 28.04", unit=(u.hourangle, u.deg), frame="icrs"),
        "obj_3c273":     SkyCoord("12 29 06.699 +02 03 08.60", unit=(u.hourangle, u.deg), frame="icrs"),
        "someother":     SkyCoord("14 35 42.8685615528 +40 18 02.133470196", unit=(u.hourangle, u.deg), frame="icrs"),
        "CDF_South":     SkyCoord("53.161 -27.791", unit=u.deg, frame="icrs"),
    })

    radius_arcmin: u.Quantity = 5.0 * u.arcmin

    max_records: int = 3

    def label_csv_path(self) -> Path:
        p = Path(self.label_csv).expanduser()
        return p if p.is_absolute() else self.dataset_dir / p

    @classmethod
    def clean_path(cls, value: str) -> Path:
        return Path(value.strip().strip('"').strip("'"))

    @classmethod
    def clean_str(cls, value: str) -> str:
        return value.strip().strip('"').strip("'")
    
    # -------- Get config from env 
    @classmethod
    def from_env(cls) -> "AstroosConfig":
        def env(key: str, default: Optional[str] = None) -> str:
            v = os.getenv(key, default)
            if v is None:
                raise RuntimeError(f"Missing required env var: {key}")
            return v

        return cls(
            dataset_dir=cls.clean_path(env("PIPELINE_DATASET_DIR")).expanduser(),
            dataset_name=cls.clean_str(env("PIPELINE_DATASET_NAME")),
            pipeline_dir=Path(env("PIPELINE_DIR")).expanduser(),
            pipeline_name=env("PIPELINE_NAME"),
            pipeline_minor_version=int(env("PIPELINE_MINOR_VERSION")),
            label_def_file=cls.clean_str(env("PIPELINE_LABEL_DEF_CSV")),
        )

    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="Astroos Configuration",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # pytest query file (optional)
        p.add_argument("-q", type=Path, help="pytest query file")

        # core paths / identity
        p.add_argument("--dataset-dir", type=Path, help="Dataset root directory")
        p.add_argument("--dataset-name", type=str, help="Dataset name")
        p.add_argument("--pipeline-dir", type=Path, help="Pipeline directory")
        p.add_argument("--pipeline-name", type=str, help="Pipeline name")
        p.add_argument("--pipeline-minor-version", type=int, help="Minor version")
        p.add_argument("--label-def-file", type=str, help="Label definition CSV file")
        p.add_argument("--max-records", type=int, default=3, help="Max records to fetch from query")

        # runtime overrides
        p.add_argument(
            "--target",
            choices=list(cls().coords.keys()) if False else None,
            help="Target key from coords table",
        )
        p.add_argument(
            "--radius-arcmin",
            type=float,
            help="Override target search radius (arcmin)",
        )

        return p
    
    @classmethod
    def from_cli(cls, argv: Optional[list[str]] = None) -> "AstroosConfig":
        parser = cls.build_arg_parser()
        args = parser.parse_args(argv)

        # start from env
        base = cls.from_env()

        # merge CLI → env
        cfg = cls(
            dataset_dir=args.dataset_dir or base.dataset_dir,
            dataset_name=args.dataset_name or base.dataset_name,
            pipeline_dir=args.pipeline_dir or base.pipeline_dir,
            pipeline_name=args.pipeline_name or base.pipeline_name,
            pipeline_minor_version=(
                args.pipeline_minor_version
                if args.pipeline_minor_version is not None
                else base.pipeline_minor_version
            ),
            label_def_file=args.label_def_file or base.label_def_file,
            coords=base.coords,  # unchanged for now
            max_records=args.max_records if args.max_records is not None else base.max_records,
        )

        # optional runtime coord override
        if args.target:
            if args.target not in cfg.coords:
                raise ValueError(
                    f"Unknown target '{args.target}'. "
                    f"Valid targets: {list(cfg.coords)}"
                )

            if args.radius_arcmin is not None:
                spec = cfg.coords[args.target]
                cfg = cls(
                    **{**cfg.__dict__,
                    "coords": {
                        **cfg.coords,
                        args.target: CoordSpec(
                            value=spec.value,
                            fmt=spec.fmt,
                            radius_arcmin=args.radius_arcmin,
                        ),
                    }}
                )

        return cfg

    def __repr__(self):
        rows = [
            ("dataset_dir",            self.dataset_dir),
            ("dataset_name",           self.dataset_name),
            ("pipeline_dir",           self.pipeline_dir),
            ("pipeline_name",          self.pipeline_name),
            ("pipeline_minor_version", self.pipeline_minor_version),
            ("label_def_file",         self.label_def_file),
            ("coords",                 f"{len(self.coords)} entries"),
            ("max_records",            self.max_records),
        ]
        return ascii_kv_table(rows, title="AstroosConfig")
    


