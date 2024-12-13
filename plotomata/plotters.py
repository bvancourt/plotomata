"""
This module contains the user-facing plotting functions for plotomata package.
They are sort of wrapper functions for another module, _strict_plotters, which
does the actual plotting, but requires input in rigidly defines formats, whereas
the plotters functions interpret a flexible "language" of arguments.
"""

import importlib
import os

# Load plotomata component modules
try:
    from . import style_packets, color_palettes, _utils

    from ._utils import PassthroughDict
    from .style_packets import StylePacket
    from .color_palettes import Color

except ImportError as ie:
    # Alternative import style for non-standard import.
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import style_packets, color_palettes, _utils

        from _utils import PassthroughDict
        from style_packets import StylePacket
        from color_palettes import Color

    except ImportError as _:
        raise ImportError(
            "plotomata failed to import component modules."
        ) from ie

    except Exception as e:
        raise ImportError(
            "Unexpected error while importing plotomata component modules."
        ) from e

except Exception as e:
    raise ImportError from e

from typing import TypeAlias, Callable
import warnings

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap, Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, Size
import seaborn as sns
import pandas as pd
from pandas.core.series import Series

# reloads internal dependencies on reload
importlib.reload(color_palettes)
importlib.reload(style_packets)
importlib.reload(_utils)


def _stacked_bars(data_df, style_packet, ax) -> Axes:
    return ax


def _color_legend(ax) -> Axes:
    return ax


def _size_legend():
    pass
