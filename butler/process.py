import pandas as pd

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    import lsst.geom as geom
    rsp_mode = True
except ImportError:
    pass

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

BANDS = ["g", "r", "i", "z", "y"]

# cutout stamp size (pixels)
STAMP_W = 100
STAMP_H = 100

def worker_patch(args):
    tract, patch, object_rows = args

    from lsst.daf.butler import Butler
    butler = Butler("dp1", collections="LSSTComCam/DP1")

    # Load each band ONCE per patch
    coadds = {
        b: butler.get("deep_coadd", tract=tract, patch=patch, band=b)
        for b in BANDS
    }

    ext = geom.Extent2I(STAMP_W, STAMP_H)

    for row in object_rows:
        ra_deg = float(row["coord_ra"])
        dec_deg = float(row["coord_dec"])

        # cross-match with HST


        # SpherePoint expects (lon, lat) as Angles.
        # Use degrees explicitly.
        sky = geom.SpherePoint(ra_deg * geom.degrees, dec_deg * geom.degrees)

        for band, exp in coadds.items():
            wcs = exp.getWcs()
            if wcs is None:
                # Shouldn't happen for coadds, but guard anyway
                continue

            # Convert sky coordinate to pixel coordinate in this exposure
            pix = wcs.skyToPixel(sky)  # returns lsst.geom.Point2D

            # Optional: skip objects whose pixel center is off-image
            bbox = exp.getBBox()
            if not bbox.contains(geom.Point2I(int(round(pix.getX())), int(round(pix.getY())))):
                continue

            cutout = exp.getCutout(pix, ext)

            # Write FITS directly by filename (don't open a file handle yourself)
            out = f"cutout_t{tract}_p{patch}_obj{int(row['objectId'])}_{band}.fits"
            cutout.writeFits(out)

    return len(object_rows)

def build_groups(objects):
    groups = defaultdict(list)
    for row in objects:
        groups[(int(row["tract"]), int(row["patch"]))].append(row)
    return [(t, p, rows) for (t, p), rows in groups.items()]

query = """
SELECT TOP {max_records}
  objectId,
  tract,
  patch,
  coord_ra,
  coord_dec,
  refExtendedness
FROM dp1.Object
WHERE coord_ra BETWEEN {ra_min} AND {ra_max}
  AND coord_dec BETWEEN {dec_min} AND {dec_max}
"""

# ECDFS-ish region
query = query.format(
    max_records=10,
    ra_min=52,
    ra_max=53,
    dec_min=-28,
    dec_max=-27,
)

tap_service = get_tap_service("tap")
res = tap_service.search(query)
objects = res.to_table()
print(f"{len(objects)} returned")

tasks = build_groups(objects)

with ProcessPoolExecutor(max_workers=8) as ex:
    for _ in ex.map(worker_patch, tasks):
        pass
