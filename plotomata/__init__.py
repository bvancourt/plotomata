"""
Plotomata is a package for automating generation of publication style/quality 
plots. At this stage in development, it's use is not recommended, but it will be 
great eventually, maybe.
"""

import importlib
import plotomata.plotters
import plotomata.colors

importlib.reload(plotomata.plotters)
importlib.reload(plotomata.colors)

from .plotters import bar_plot, column_plot
from .colors import tab20_colors, nb50_colors

__version__ = "0.0.0"
