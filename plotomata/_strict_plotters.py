import importlib

try:
    from . import color_palettes, style_packets

    importlib.reload(color_palettes)
    importlib.reload(style_packets)
    from .color_palettes import Color
    from .style_packets import StylePacket
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, style_packets

        importlib.reload(color_palettes)
        importlib.reload(style_packets)
        from color_palettes import Color
        from style_packets import StylePacket
    except ImportError as ie2:
        raise ie2 from ie
except Exception as e:
    raise ImportError from e
else:
    import os

from typing import TypeAlias, Callable
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap, Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, Size
import seaborn as sns
import pandas as pd
from pandas.core.series import Series
from numpy.typing import NDArray


def _strict_bar_plot(
    ax_in: Axes,  # axes to put the bars on
    bar_heights: np.ndarray,  # shape = (n_layers, n_columns), dtype = np.float64
    colors: list[list[Color]],  # "shape" must match bar_heights
    style_packet: StylePacket,
    mode: str = "stacked",
    direction: str = "vertical",
) -> tuple[Axes, list[tuple[float, float]]]:  # axes and error bar positions
    """ """
    # assert valid inputs
    assert isinstance(ax_in, Axes)
    assert mode in {"stacked", "packed"}
    assert direction in {"vertical", "horizontal"}
    assert bar_heights.dtype is np.float64
    assert len(colors) == bar_heights.shape[0]
    for colors_elem in colors:
        assert len(colors_elem) == bar_heights.shape[1]
        for color in colors_elem:
            assert isinstance(color, Color)


def _error_bar(
    ax: Axes,
    base_coords: tuple[float, float],
    length: float,
    direction: str = "up",
    mode: str = "stick",
) -> Axes:
    assert isinstance(ax, Axes)
    assert isinstance(base_coords, tuple)
    assert len(base_coords) == 2
    assert isinstance(base_coords[0])
    assert mode in {"stick", "plunger"}
    assert isinstance(length, float)
    assert length >= 0
    assert direction in {
        "up",
        "down",
        "left",
        "right",
        "vertical",
        "horizontal",
    }
