# Purpose
`wakai` is codebase developed for age-dating host stars of transiting planets.
It is mainly a wrapper to run scripts in different repos under ../ext_tools/.
The scripts should be installed in .venv in their respective directories using uv.
Do not use conda environment.

## Internal Code
* the `wakai` package (./wakai/) is split by responsibility: star.py, planet.py,
  cluster.py, iso.py, sed.py, vizier.py, catalog_keys.py, database.py, reference.py
* ./wakai/catalog.py is a compatibility facade that re-exports from those modules
  for existing notebooks -- it's already refactored, not a file to review/change
* use the focused modules as inspiration and point out bugs or optimizations when necessary
* ask permission before adding new functions to any of these modules

## External code
* several packages are needed to run some notebooks such as gyro-interp, eagles, etc.
* these packages are available as conda environments in /ut3/jerome/miniconda3/envs/

## Plotting
* Use rcPrams.txt as matplotlib configuration when plotting figures
* use dpi=200
* use png instead of pdf file output
* all plotting outputs should live in ./figures/

## Environment
* wakai conda environment is available at /ut3/jerome/miniconda3/envs/wakai
* use pip install only if necessary

## Data files
* ./data hosts large data files. Ask permission first before reading them.
* Do not read large .csv files. If necessary, use head or tail first to check file format.
