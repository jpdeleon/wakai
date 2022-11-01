import itertools
import matplotlib.pyplot as pl
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))

def get_relative_err_index(df, par='st_age', rel_err=0.1):
    return df.apply(lambda x: ((x[par+'err1']/x[par]<=rel_err) \
                                    & (x[par+'err2']/x[par]<=rel_err)), 
                                    axis=1)

def plot_planets(df,
                x='logTeff', y='st_rotp', 
                ax=None,
                plot_kwds={'c': 'C0'}
                ):
    if ax is None:
        fig, ax = pl.subplots(dpi=150)
    df.plot.scatter(x=x, y=y, ax=ax, **plot_kwds)
    return ax

def get_nexsci_data(table_name="ps", clobber=False):
    """
    ps: self-consistent set of parameters
    pscomppars: #a more complete, though not necessarily self-consistent set of parameters
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
    print("Column definitions: ", url)
    fp = f"../data/nexsci_{table_name}.csv"
    if clobber:
        nexsci_tab = NasaExoplanetArchive.query_criteria(table=table_name, where="discoverymethod like 'Transit'")
        df_nexsci = nexsci_tab.to_pandas()
        df_nexsci.to_csv(fp, index=False)
        print("Saved: ", fp)
    else:
        df_nexsci = pd.read_csv(fp)
        print("Loaded: ", fp)
    return df_nexsci