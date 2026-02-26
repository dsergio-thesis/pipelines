
from astropy.coordinates import SkyCoord
from astropy import units as u

c1 = SkyCoord(ra=52.95120313 * u.deg, dec=-27.52756084 * u.deg, frame='icrs')

c2 = SkyCoord(ra=53.00185394 * u.deg, dec=-27.68044472 * u.deg, frame='icrs')

sep = c1.separation(c2).to(u.arcsec)

print(sep)
