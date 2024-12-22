"""
Code to manage figures.
"""

import importlib

try:
    from . import style_packets

    importlib.reload(style_packets)
    from .style_packets import StylePacket, SettingsPacket

except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import style_packets

        importlib.reload(style_packets)

        from style_packets import StylePacket, SettingsPacket

    except ImportError as ie2:
        raise ie2 from ie

except Exception as e:
    raise ImportError from e

else:
    import os


from enum import Enum, auto
import matplotlib.pyplot as plt
import numpy as np


class ElementContraintTypes(Enum):
    MATCH_HEIGHT = auto()
    MATCH_WIDTH = auto()
    ALIGN_VERTICALLY = auto()
    ALIGN_HORIZONTALLY = auto()
    ALIGN_TOP = auto()
    ALIGN_BOTTOM = auto()
    ALIGN_LEFT = auto()
    ALIGN_RIGHT = auto()
    JOIN_LEFT = auto()
    JOIN_RIGHT = auto()
    JOIN_BOTTOM = auto()
    JOIN_TOP = auto()


class FigureElement:
    """
    This represents a rectangular area of a figure with something on it. It is
    a base class for AxesElement and NoAxesElement below, not intended to be
    used directly.
    """

    pass


class AxesElement(FigureElement):
    """
    This could represent axes used for actual plotting or peripheral elements
    such as legends and colorbars, which will also get matplotlib Axes.
    """

    pass


class NoAxesElement(FigureElement):
    """
    This represents the space on a Figure taken up by ticks and axis lables.
    """

    pass


class HyperFig:
    """
    This class represents a sort of generalized version of a matplotlib figure,
    which could have one Figure with multiple Axes or multiple connected
    Figures. For example, it may or may not be best to save legends, colorbars,
    etc. to separate image files from the plot they refer to. Similarly, several
    plots with matching axes could be displayed as subplots or individual image
    files. Using this should make it very easy to switch between those options.
    """

    def __init__(
        self,
        elements: dict[str, AxesElement | NoAxesElement],
        constraints: list[tuple[ElementContraintTypes, str, str]],
    ):
        pass

    def make_figs(
        self,
        style_packet: SettingsPacket,
        settings_packet: SettingsPacket,
    ):
        pass
