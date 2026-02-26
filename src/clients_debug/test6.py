from lsst.daf.butler import Butler

butler = Butler("dp1")

image = butler.get(
    "deepCoadd",
    tract=3828,
    patch=23,
    band="r"
)
