
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple
import os

import astropy.units as u
from astropy.coordinates import SkyCoord

CoordFormat = Literal["hmsdms", "deg"]


@dataclass(frozen=True)
class CoordSpec:
    value: str
    fmt: CoordFormat = "hmsdms"
    radius_arcmin: float = 5.0


@dataclass(frozen=True)
class PipelineConfig:
    dataset_dir: Path
    dataset_name: str
    pipeline_dir: Path
    pipeline_name: str
    pipeline_minor_version: int
    label_csv: str

    frame: str = "icrs"
    obstime: Optional[str] = None
    equinox: Optional[str] = None

    coords: Dict[str, CoordSpec] = field(default_factory=lambda: {
        "virgo_cluster": CoordSpec("12 30 49.423 +12 23 28.04", "hmsdms", 30.0),
        "obj_3c273":     CoordSpec("12 29 06.699 +02 03 08.60", "hmsdms", 2.0),
        "someother":     CoordSpec("14 35 42.8685615528 +40 18 02.133470196", "hmsdms", 3.0),
        "CDF_South":     CoordSpec("53.161 -27.791", "deg", 10.0),
    })

    def label_csv_path(self) -> Path:
        p = Path(self.label_csv).expanduser()
        return p if p.is_absolute() else self.dataset_dir / p

    def get_target(self, key: str) -> Tuple[SkyCoord, u.Quantity]:
        spec = self.coords[key]

        kwargs = {"frame": self.frame}
        if self.obstime:
            kwargs["obstime"] = self.obstime
        if self.equinox:
            kwargs["equinox"] = self.equinox

        if spec.fmt == "hmsdms":
            coord = SkyCoord(spec.value, unit=(u.hourangle, u.deg), **kwargs)
        else:
            ra, dec = map(float, spec.value.split())
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, **kwargs)

        return coord, spec.radius_arcmin * u.arcmin

    # -------- env wiring --------
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        def env(key: str, default: Optional[str] = None) -> str:
            v = os.getenv(key, default)
            if v is None:
                raise RuntimeError(f"Missing required env var: {key}")
            return v

        return cls(
            dataset_dir=Path(env("PIPELINE_DATASET_DIR")).expanduser(),
            dataset_name=env("PIPELINE_DATASET_NAME"),
            pipeline_dir=Path(env("PIPELINE_DIR")).expanduser(),
            pipeline_name=env("PIPELINE_NAME"),
            pipeline_minor_version=int(env("PIPELINE_MINOR_VERSION")),
            label_csv=env("PIPELINE_LABEL_CSV"),
            frame=os.getenv("PIPELINE_FRAME", "icrs"),
            obstime=os.getenv("PIPELINE_OBSTIME"),
            equinox=os.getenv("PIPELINE_EQUINOX"),
        )
