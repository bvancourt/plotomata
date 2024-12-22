import importlib

try:
    from . import color_palettes, style_packets, _utils

    importlib.reload(color_palettes)
    importlib.reload(style_packets)
    importlib.reload(_utils)
    from .color_palettes import Color
    from .style_packets import StylePacket, SettingsPacket
    from ._utils import all_are_instances
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, style_packets, _utils

        importlib.reload(color_palettes)
        importlib.reload(style_packets)
        importlib.reload(_utils)
        from color_palettes import Color
        from style_packets import StylePacket, SettingsPacket
        from _utils import all_are_instances
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


def _strict_line_plot(
    ax: Axes,
    x: np.ndarray[float],
    y: np.ndarray[float],
    color: Color = Color(0, 0, 0),
) -> Axes:
    ax.plot(x, y, color=color)
    return ax


def _strict_bar_plot(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,  # axes to put the bars on
    bar_lengths: np.ndarray,  # shape = (n_layers, n_columns), dtype = np.float64
    layer_names: list[str],
    stack_names: list[str],
    colors: list[list[Color]],  # "shape" must match bar_heights
    mode: str = "stacked",
    direction: str = "vertical",
) -> tuple[Axes, list[tuple[float, float]]]:  # axes and error bar positions
    """
    Makes a bar plot on provided axes.
    """
    if not settings_packet.skip_asserts:
        # assert valid inputs (type checks)
        assert isinstance(ax, Axes)
        assert mode in {"stacked", "packed"}
        assert direction in {"vertical", "horizontal"}
        assert bar_lengths.dtype is np.float64
        assert len(colors) == bar_lengths.shape[0]
        for colors_elem in colors:
            assert len(colors_elem) == bar_lengths.shape[1]
            assert all_are_instances(colors_elem, Color)
        assert isinstance(style_packet, StylePacket)
        assert isinstance(layer_names, list)
        assert all_are_instances(layer_names, str)
        assert isinstance(stack_names, list)
        assert all_are_instances(stack_names, str)

    if direction == "vertical":
        distance_spine_to_stack_center = (
            1 + style_packet.outer_margins[0]
        ) * 0.5
        ax.set_xlim(
            -distance_spine_to_stack_center,
            bar_lengths.shape[1] + distance_spine_to_stack_center,
        )
        ax.set_xticks(np.arange(bar_lengths.shape[0]))
        _label_x_ticks(style_packet, settings_packet, ax, stack_names)

    elif direction == "horizontal":
        distance_spine_to_stack_center = (
            1 + style_packet.outer_margins[1]
        ) * 0.5
        ax.set_ylim(
            -distance_spine_to_stack_center,
            bar_lengths.shape[1] + distance_spine_to_stack_center,
        )
        ax.set_yticks(np.arange(bar_lengths.shape[0]))
        _label_y_ticks(style_packet, settings_packet, ax, stack_names)


def _block_margin_axes_dims(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    block_array_shape: tuple[int],
    block_shape: tuple[float],
) -> tuple[float, float]:
    """
    If what you want to show on some axes is an array of equal-sized "blocks"
    with blank margins between them, this will calculate the size of the axes
    from the number and dimension of the blocks and padding, measured in inches.
    """
    if not settings_packet.skip_asserts:
        assert isinstance(style_packet, StylePacket)
        assert isinstance(block_array_shape, tuple)
        assert len(block_array_shape) == 2
        assert all_are_instances(block_array_shape, int)
        for dim in block_array_shape:
            assert dim > 0
        assert len(block_shape) == 2
        assert all_are_instances(block_shape, int)
        for dim in block_shape:
            assert dim > 0

    return (
        (
            style_packet.outer_margin[0] * 2
            + style_packet.inner_margin[0] * (block_array_shape[0] - 1)
            + block_shape[0] * block_array_shape[0]
        ),
        (
            style_packet.outer_margin[1] * 2
            + style_packet.inner_margin[1] * (block_array_shape[1] - 1)
            + block_shape[1] * block_array_shape[1]
        ),
    )


def _label_x_ticks(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,
    x_tick_labels: list[str],
) -> None:
    """ """
    if not settings_packet.skip_asserts:
        assert len(ax.get_xticklabels()) == len(x_tick_labels)

    display_labels = style_packet.list_display_names(x_tick_labels)
    label_colors = style_packet.list_colors(x_tick_labels)

    ax.set_xticklabels(display_labels)

    for text_obj, color in zip(ax.get_xticklabels(), label_colors):
        text_obj.set_color(color)

    return ax


def _label_y_ticks(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,
    y_tick_labels: list[str],
) -> None:
    """
    Re-labels y ticks of an existing plot, applying colors, etc from
    StylePackets.
    """
    if not settings_packet.skip_asserts:
        assert len(ax.get_yticklabels()) == len(y_tick_labels)

    display_labels = style_packet.list_display_names(y_tick_labels)
    label_colors = style_packet.list_colors(y_tick_labels)

    ax.set_yticklabels(display_labels)

    for text_obj, color in zip(ax.get_yticklabels(), label_colors):
        text_obj.set_color(color)

    return ax


def _draw_line(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,
    x: np.ndarray[float],
    y: np.ndarray[float],
    color: Color,
):
    if not settings_packet.skip_asserts:
        assert isinstance(style_packet, StylePacket)
        assert isinstance(settings_packet, SettingsPacket)
        assert isinstance(ax, Axes)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.floating)
        assert x.shape[0] == y.shape[0]
        assert len(x.shape) == len(y.shape) == 1

    ax.plot()


def _draw_error_bar(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,
    base_coords: tuple[float, float],
    length: float,
    direction: str = "up",
    mode: str = "stick",
):
    if not settings_packet.skip_asserts:
        assert isinstance(style_packet, StylePacket)
        assert isinstance(settings_packet, SettingsPacket)
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

    match direction:
        case "up":
            x_ends = (np.array([base_coords[0], base_coords[0]]),)
            y_ends = (np.array([base_coords[1], base_coords[1] + length]),)

        case "down":
            x_ends = (np.array([base_coords[0], base_coords[0]]),)
            y_ends = (np.array([base_coords[1], base_coords[1] - length]),)

        case "left":
            x_ends = (np.array([base_coords[0], base_coords[0] - length]),)
            y_ends = (np.array([base_coords[1], base_coords[1]]),)

        case "right":
            x_ends = (np.array([base_coords[0], base_coords[0]] + length),)
            y_ends = (np.array([base_coords[1], base_coords[1]]),)

        case "vertical":
            x_ends = (np.array([base_coords[0], base_coords[0]]),)
            y_ends = (
                np.array([base_coords[1] - length, base_coords[1] + length]),
            )

        case "horizontal":
            x_ends = (
                np.array([base_coords[0] - length, base_coords[0] + length]),
            )
            y_ends = (np.array([base_coords[1], base_coords[1]]),)

    return _draw_line(
        style_packet,
        settings_packet,
        ax,
        x_ends,
        y_ends,
    )


def _strict_clustermap(
    style_packet: StylePacket,
    settings_packet: SettingsPacket,
    ax: Axes,
):
    pass
