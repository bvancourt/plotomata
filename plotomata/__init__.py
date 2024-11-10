# This first bit of code makes reloading the package reload each module.
import importlib
import plotomata.plotters
import plotomata.colors

importlib.reload(plotomata.plotters)
importlib.reload(plotomata.colors)

from .plotters import bar_plot, column_plot
from .colors import tab20_colors, nb50_colors

__version__ = "0.0.0"
