import os
import sys
import matplotlib.pyplot as pl

sys.path.insert(0, '../code')
assert os.environ["CONDA_DEFAULT_ENV"] == 'wakai'

plotpar = {'axes.labelsize': 16,
           'font.size': 16,
           'legend.fontsize': 16,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           #'text.usetex': False,
           'xtick.direction': 'in',
           'ytick.direction': 'in'
           }
pl.rcParams.update(plotpar)
