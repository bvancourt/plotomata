"""
Plotomata is a package for automating generation of publication style/quality
plots. At this stage in development, it's use is not recommended, but it will be
great eventually, maybe.
"""

from . import plotters
from . import color_sets
import importlib

importlib.reload(plotters)
importlib.reload(color_sets)

from .plotters import bar_plot, column_plot, categorical_scatter
from .color_sets import tab20_colors, nb50_colors

__version__ = "0.0.0"
