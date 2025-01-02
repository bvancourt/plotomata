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


from dataclasses import dataclass
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


@dataclass
class FigureElement:
    """
    This represents a rectangular area of a figure with something on it. It is
    a base class for AxesElement and NoAxesElement below, not intended to be
    used directly.
    """

    width: float
    height: float
    shape_is_fixed: bool
    depends_on: list[str] | None


@dataclass
class PlotElement(FigureElement):
    """
    This could represent axes used for actual plotting or peripheral elements
    such as legends and colorbars, which will also get matplotlib Axes.
    """

    pass


@dataclass
class LegendElement(FigureElement):
    """
    This represents a legend, colorbar, or similar peripheral bit of image
    associated with a plot element.
    """

    pass


@dataclass
class NoAxesElement(FigureElement):
    """
    This represents the space on a Figure taken up by ticks and axis lables.
    """

    pass


class InvalidHyperfig(Exception):
    pass


class Hyperfig:
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
        elements: dict[str, PlotElement | LegendElement | NoAxesElement],
        constraints: list[tuple[ElementContraintTypes, str, str]],
        settings_packet: SettingsPacket,
    ):
        self.elements = elements
        self.constraints = constraints
        self.settings_packet = settings_packet

    def make_figs(
        self,
        style_packet: SettingsPacket,
    ):
        pass

    def assert_validity(self):
        for name, element in self.elements.items():
            if element.depends_on is not None:
                for other_name in element.depends_on:
                    if not other_name in self.elements:
                        raise InvalidHyperfig(
                            f"element {name} depends on {other_name}, "
                            + "which was not found in self.elements.\n"
                        )

        for constrain_type, subject_name, object_name in self.constraints:
            if not subject_name in self.elements:
                raise InvalidHyperfig(
                    f"{constrain_type} constraint depends on {subject_name}, "
                    + "subject which was not found in self.elements.\n"
                )

            if not object_name in self.elements:
                raise InvalidHyperfig(
                    f"{constrain_type} constraint depends on {object_name}, "
                    + "object which was not found in self.elements.\n"
                )

            assert isinstance(constrain_type, ElementContraintTypes)

            self.settings_packet.assert_validity()
