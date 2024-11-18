"""
Plotomata is a package for automating generation of publication style/quality
plots. At this stage in development, it's use is not recommended, but it will be
great eventually.
"""

from . import plotters
from . import color_sets
from . import style_packets
from . import utils
import importlib

importlib.reload(plotters)
importlib.reload(color_sets)
importlib.reload(style_packets)
importlib.reload(utils)

from .plotters import bar_plot, column_plot, scatter_plot
from .color_sets import tab20_colors, nb50_colors
from .style_packets import StylePacket
from .utils import PassthroughDict

__version__ = "0.0.1"
