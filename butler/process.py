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
        cutout = coadds[band].getCutout(geom.SpherePoint(0, 0), 100)  # example cutout
        with open(f"cutout_{tract}_{patch}_{band}.fits", "wb") as f:
            cutout.writeFits(f)

    return len(object_rows)

def build_groups(objects):
    groups = defaultdict(list)
    for row in objects:
        groups[(row.tract, row.patch)].append(row)
    return [(t, p, rows) for (t, p), rows in groups.items()]

# objects should already be preselected (TAP recommended) with tract/patch + coords + objectId
tap_service = get_tap_service("tap")
query = 
"""
SELECT objectId, tract, patch
FROM dp1.object
WHERE 1=1
-- AND extendedness > 0.5
AND coord_ra IS BETWEEN 52 AND 53
AND coord_dec IS BETWEEN -28 AND -27
LIMIT 10
"""
res = self.tap_service.search(query)
objects = res.to_table()

tasks = build_groups(objects)

with ProcessPoolExecutor(max_workers=8) as ex:
    for n in ex.map(worker_patch, tasks):
        pass
