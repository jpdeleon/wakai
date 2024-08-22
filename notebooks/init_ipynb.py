import os
import sys
import matplotlib.pyplot as pl

sys.path.insert(0, '../code')
assert os.environ["CONDA_DEFAULT_ENV"] == 'wakai'

plotpar = {
           'font.size': 20,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'xtick.direction': 'in',
           'ytick.direction': 'in',
           'xtick.color':'black',
           'ytick.color':'black',
           'xtick.major.width':3,
           'ytick.major.width':3,
           'xtick.major.size':10,
           'ytick.major.size':10,
           'xtick.minor.width':1,
           'ytick.minor.width':1,
           'xtick.minor.size':6,
           'ytick.minor.size':6,
           'axes.labelsize': 16,
           'axes.labelcolor':'black',
           'axes.labelcolor':'black',
           'axes.spines.top':True,
           'axes.spines.right':True,
           'axes.linewidth':3,
           'axes.edgecolor':'black',
           'figure.facecolor':'none',
           'legend.facecolor':'none',
           'legend.fontsize': 16,
           'text.color':'black',
           'pdf.fonttype': 42,
           #'text.usetex': False,
           }
pl.rcParams.update(plotpar)