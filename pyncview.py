#!/usr/bin/env python3
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as anim
# import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as tick
import plot as ncplot
import filters as fl
import argparse

parser = argparse.ArgumentParser(description="Simple NetCDF file viewer in Python for 3D fields in a sequence of snapshots. Select .nc file, a variable and a snapshot")
parser.add_argument("filename", help="NetCDF file name", type=str)
parser.add_argument("varname", help="Variable name. Must exist in file", type=str)
parser.add_argument("snap", help="Time snapshot index", type=int)
parser.add_argument("--latitude", help="Plot a vertical section at latitude index", type=int)
parser.add_argument("--level", help="Plot a horizontal section at level index", type=int)
parser.add_argument("--invertl", help="Invert level axis", action="store_true")
args = parser.parse_args()

ds = nc.Dataset(args.filename)
var = ds.variables[args.varname][args.snap,:,:,:]
mask = np.isnan(var)
if args.latitude:
    lev_factor = -1 if args.invertl else 1
    ncplot.plot_lonlev(ds, var, args.varname, args.latitude, mask=mask, lev_factor=lev_factor)
elif args.level:
    ncplot.plot_lonlat(ds, var, args.varname, args.level, mask=mask)