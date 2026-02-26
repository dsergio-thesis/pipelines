import pandas as pd

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    from lsst.rsp.utils import get_pyvo_auth
    import lsst.geom as geom

    # other LSST dependencies
    from pyvo.dal.adhoc import DatalinkResults
    from astropy.time import Time
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery

    rsp_mode = True
except ImportError:
    pass


from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

BANDS = ["g","r","i","z","y"]

def worker_patch(args):
    tract, patch, object_rows = args

    # print(f"starting work with {args}")

    # Create Butler inside process
    from lsst.daf.butler import Butler
    butler = Butler("dp1", collections="LSSTComCam/DP1")  # common DP1 collection

    # Load each band ONCE
    coadds = {
        b: butler.get("deep_coadd", tract=tract, patch=patch, band=b)
        for b in BANDS
    }

    # For each object: cut out locally, write immediately (don’t accumulate)
    # pseudo:
    # for row in object_rows:
    #   center = ...
    #   stamp_stack = np.stack([cut(coadds[b], center) for b in BANDS])
    #   append_to_zarr_or_memmap(stamp_stack, row.objectId, ...)
    
    for band in BANDS:
        lat = geom.Angle(0.1)
        long = geom.Angle(0.1)
        ext = geom.Extent2I(100, 100)
        cutout = coadds[band].getCutout(geom.SpherePoint(lat, long), ext)
        with open(f"cutout_{tract}_{patch}_{band}.fits", "wb") as f:
            cutout.writeFits(f)

    return len(object_rows)

def build_groups(objects):
    groups = defaultdict(list)
    print(f"objects: {objects} type: {type(objects)}")
    for row in objects:
        # print(f"row: {row}, type: {type(row)}")
        groups[(row['tract'], row['patch'])].append(row)
    return [(t, p, rows) for (t, p), rows in groups.items()]

query = \
"""
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

# Extended Chandra Deep Field South (ECDFS)
query = query.format(
        max_records=10,
        ra_min=52,
        ra_max=53,
        dec_min=-28,
        dec_max=-27,
        )
print(f"query: \n {query}")
tap_service = get_tap_service("tap")
res = tap_service.search(query)
objects = res.to_table()
print(f"{len(objects)} returned")

tasks = build_groups(objects)

with ProcessPoolExecutor(max_workers=8) as ex:
    for n in ex.map(worker_patch, tasks):
        pass
