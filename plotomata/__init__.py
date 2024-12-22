"""
Plotomata provides tools for creating collections of high-quality, consistently
formatted plots (e.g. for a scientific paper).
"""

from . import plotters
from . import color_palettes
from . import style_packets
from . import legacy_plotters

import importlib

importlib.reload(plotters)
importlib.reload(color_palettes)
importlib.reload(style_packets)
importlib.reload(legacy_plotters)

# from .plotters import bar_plot, column_plot, scatter_plot
from .color_palettes import Color, nb50_colors, tab20_colors
from .style_packets import StylePacket, SettingsPacket
from .legacy_plotters import (
    legacy_scatter_plot,
    legacy_bar_plot,
    legacy_column_plot,
)

__version__ = "0.0.2"
