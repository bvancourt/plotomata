"""
The purpose of this file is to allow effectively importing and repeatedly
reloading the package in R using

reticulate::source_python("/path/to/plotomata/reticulate_source.py")

This is a development/debugging feature only needed to repeatedly modify 
plotomata code and test it in R.
"""

import os
import sys
import importlib

sys.path.insert(
    0, os.path.join(os.path.split(os.path.abspath(__file__))[0], "plotomata")
)

import plotters
import color_sets

importlib.reload(plotters)
importlib.reload(color_sets)

from plotters import *
from color_sets import *
