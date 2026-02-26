
# Select Objects
```
SELECT objectId, tract, patch
FROM dp1.object
WHERE extendedness > 0.5
LIMIT 100000
```

# Group Patches
```
from collections import defaultdict

patch_groups = defaultdict(list)

for obj in objects:
    key = (obj.tract, obj.patch)
    patch_groups[key].append(obj)
```

# Get each coadd
```
coadd = butler.get(
    "deepCoadd",
    tract=tract,
    patch=patch,
    band="r"
)
```

# Get Cutout
```
from lsst.geom import Point2D
from lsst.afw.image import ExposureF

cutout = coadd.getCutout(center, size)
```

# Bands
```
bands = ["g","r","i","z","y"]

images = {
    b: butler.get("deepCoadd", tract=t, patch=p, band=b)
    for b in bands
}
```

