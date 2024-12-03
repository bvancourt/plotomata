"""
Plotomata provides tools for creating collections of high-quality, consistently
formatted plots (e.g. for a scientific paper). At this stage in development, 
it use is not recommended, but it will eventualy be great.
"""

from . import plotters
from . import color_palettes
from . import style_packets

import importlib

importlib.reload(plotters)
importlib.reload(color_palettes)
importlib.reload(style_packets)

# from .plotters import bar_plot, column_plot, scatter_plot
from .color_palettes import Color
from .style_packets import StylePacket

__version__ = "0.0.2"
