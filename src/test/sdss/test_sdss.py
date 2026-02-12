
import importlib
from astropy.table import Table
import sys
from astroos_pipelines.sdss.query import AstroosQuerySDSS
importlib.reload(sys.modules['astroos_pipelines.sdss.query'])

def test_astroosquery_not_None():
    query = AstroosQuerySDSS(
            root_dir = 'test_data'
            )
    assert query is not None

def test_astroosquery_basic_ADQL():

    query = AstroosQuerySDSS(
            root_dir = 'test_data'
            )
    adql = \
    """
    SELECT TOP 10
        p.objid, p.ra, p.dec, s.z, s.class
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON p.objid = s.bestobjid
    WHERE s.z > 0.1
    """
    result = query.query_skyserver(adql)

    assert result is not None
    assert len(result) == 10
    assert 'objid' in result.columns
    assert 'ra' in result.columns
    assert 'dec' in result.columns
    assert 'z' in result.columns
    assert 'class' in result.columns

    assert result is not None
    # assert isinstance(result, Table)
