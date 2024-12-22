"""
This module contains (adapted) plotting functions from plotomata 0.0.1,
before _arg_parser and StylePacket were introduced. This could be viewed as an
alternative, less experimental interface, but limited to only a few types of
plot.
"""

import importlib

try:
    from . import color_palettes, _utils, _legacy_plotter_components

    importlib.reload(color_palettes)
    importlib.reload(_utils)
    importlib.reload(_legacy_plotter_components)

    from .color_palettes import tab20_colors, nb50_colors, Color, PossibleColor
    from ._utils import PassthroughDict
    from ._legacy_plotter_components import (
        Arrayable,
        label_axes_etc,
        surrogate_color_legend_info,
        surrogate_size_legend_info_automatic,
        standardize_arrayable,
        labels_from_data,
    )
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils, _legacy_plotter_components

        importlib.reload(color_palettes)
        importlib.reload(_utils)
        importlib.reload(_legacy_plotter_components)

        from color_palettes import (
            tab20_colors,
            nb50_colors,
            Color,
            PossibleColor,
        )
        from _utils import PassthroughDict
        from _legacy_plotter_components import (
            Arrayable,
            label_axes_etc,
            surrogate_color_legend_info,
            surrogate_size_legend_info_automatic,
            standardize_arrayable,
            labels_from_data,
        )
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


_ShadedRangeType = (
    None
    | list[
        tuple[float, float] | dict[str, tuple[float, float]]
    ]  # list of below
    | tuple[float, float]  # all columns
    | dict[str, tuple[float, float]]  # per column
)


def legacy_bar_plot(
    data: pd.DataFrame,
    colors: None | str | dict[str, Color | PossibleColor] = None,
    col_colors: None | str | dict[str, Color | PossibleColor] = None,
    mode: str = "stacked",
    disp_names: dict[str, str] | None = None,
    column_order: list[int] | NDArray | None = None,
    row_order: list[int] | NDArray | None = None,
    item_width: float = 0.5,
    margin: float = 0.1,
    ax_height: float = 2.5,
    dpi: int = 300,
    edge_color: Color | PossibleColor = (0, 0, 0, 1),
    edge_width: float = 0.5,
    rotate_labels: bool | str = "auto",
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    y_label: str | None = None,
    title: str | None = None,
    spine_mode: str = "open",  # could also be "tight" or "all"
    show: bool = True,
    return_ax: bool = False,  # overrides show
    ax_in: None | Axes = None,
    **kwargs: dict[str, str | float | None],
) -> None | Axes:
    """
    This makes a stacked bar chart or grouped bar chart.

    feature ideas:
    - error bars
    - allow other input types for data
    - allow horizontal
    """
    pad_factor: float = 1.25  # demoted from kwargs

    data_df = data  # for future support of other data formats

    if isinstance(disp_names, dict):
        disp_names_dict = PassthroughDict({})
        for key in list(data_df.index) + list(data_df.columns):
            if key in disp_names:
                disp_names_dict[key] = disp_names[key]
            else:  # pass through on miss
                disp_names_dict[key] = key
    elif (disp_names is None) or (disp_names is False):
        disp_names_dict = PassthroughDict({})
    else:
        raise TypeError(
            f"disp_names should be of type dict, not {type(disp_names)}.\n"
        )

    # re-order rows and columns
    if column_order is None:
        col_list = [str(key) for key in data_df.columns]
    else:
        col_list = [str(data_df.columns[i_key]) for i_key in column_order]

    if row_order is None:
        row_list = [str(key) for key in data_df.index]
    else:
        row_list = [str(data_df.index[i_row]) for i_row in row_order]

    # Make dictionary of bar colors
    if colors == "tab20":
        colors_dict = {
            str(key): tab20_colors[i] for i, key in enumerate(row_list)
        }  # these would be the deault colors in Matplotlib

    elif (colors is None) or (colors == False) or (colors == "nb50"):
        colors_dict = {
            str(key): nb50_colors[i] for i, key in enumerate(row_list)
        }
    elif isinstance(colors, dict):
        colors_dict = {}
        default_colors = nb50_colors
        default_colors_index = 0
        for key in row_list:
            if key in colors:
                colors_dict[key] = tuple(colors[key])
            else:  # pass through on miss
                colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1
    else:
        raise ValueError(f"colors={colors} not supported.\n")

    # make dictionary of column colors (i.e. for x tick labels)
    if col_colors == "tab20":
        col_colors_dict = {
            str(key): tab20_colors[i] for i, key in enumerate(col_list)
        }  # default is like Matplotlib
    elif col_colors == "nb50":
        col_colors_dict = {
            str(key): nb50_colors[i] for i, key in enumerate(col_list)
        }
    elif isinstance(col_colors, dict):
        col_colors_dict = {}
        default_colors = tab20_colors
        default_colors_index = 0
        for key in col_list:
            if key in col_colors:
                col_colors_dict[key] = tuple(col_colors[key])
            else:
                col_colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1

    elif (col_colors is None) or (col_colors == False):
        col_colors_dict = {
            str(key): (0, 0, 0, 1) for key in col_list  # default: all black
        }
    elif col_colors == "from_colors":
        if (colors is None) or (colors == False):
            col_colors_dict = {
                str(key): (0, 0, 0, 1) for key in col_list  # default: all black
            }
        else:
            col_colors_dict = {}
            for key in col_list:
                if key in colors_dict:
                    col_colors_dict[key] = colors_dict[key]
                else:
                    col_colors_dict[key] = (0, 0, 0, 1)

    else:
        raise ValueError(f"col_colors={col_colors} not supported.\n")

    # Make matplotlib figure around axes of specified size.
    ax_position = (
        0.5 / pad_factor,
        0.5 / pad_factor,
        1 - 0.5 / pad_factor,
        1 - 0.5 / pad_factor,
    )
    fig_size = (len(col_list) * item_width * pad_factor, ax_height * pad_factor)
    if ax_in is None:
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes(ax_position)
    elif isinstance(ax_in, Axes):
        ax = ax_in
        fig = ax.get_figure()
        if not isinstance(fig, Figure):
            raise ValueError(
                f"Failed to get_figure from provided Axes ax_in={ax_in}.\n"
            )
        fig.set_size_inches(*fig_size)  # type: ignore
        ax.set_position(ax_position)
    else:
        raise TypeError(
            f"ax_in must be None or matplotlib.axes._axes.Axes, not {ax_in}.\n"
        )

    col_offsets = np.arange(len(col_list)) * (item_width + margin)
    max_bar_height = 0

    plt_bar_kwargs = {"alpha"}

    if mode == "stacked":
        bottoms = [0 for _ in col_list]

        for i_layer, label in enumerate(data_df.index):
            heights = [data_df[col_key].iloc[i_layer] for col_key in col_list]
            ax.bar(
                col_offsets,
                heights,
                bottom=bottoms,
                label=disp_names_dict[label],
                edgecolor=edge_color,
                linewidth=edge_width,
                width=item_width,
                color=colors_dict[label],
                **{
                    kwarg_key: kwarg_val
                    for kwarg_key, kwarg_val in kwargs.items()
                    if kwarg_key in plt_bar_kwargs
                },  # type: ignore
            )
            bottoms = [
                height + bottom for height, bottom in zip(heights, bottoms)
            ]

            max_bar_height = np.maximum(max_bar_height, np.max(bottoms))

    elif mode == "packed":
        sub_col_offsets = (
            (np.arange(len(row_list)) - (len(row_list) - 1) / 2)
            / len(row_list)
            * item_width
        )

        for i_layer, label in enumerate(data_df.index):
            heights = [data_df[col_key][i_layer] for col_key in col_list]
            ax.bar(
                col_offsets + sub_col_offsets[i_layer],
                heights,
                label=disp_names_dict[label],
                edgecolor=edge_color,
                linewidth=edge_width,
                width=item_width / len(row_list),
                color=colors_dict[label],
                **{
                    kwarg_key: kwargs[kwarg_key]
                    for kwarg_key in kwargs
                    if kwarg_key in plt_bar_kwargs
                },  # type: ignore
            )

            max_bar_height = np.maximum(max_bar_height, np.max(heights))
    else:
        raise ValueError(f'mode={mode} not supported; try mode="stacked".\n')

    ax.set_ylim((0, max_bar_height * (1 + margin / ax_height)))

    ax.set_xticks(col_offsets)
    ax.set_xticklabels(disp_names_dict[key] for key in col_list)

    ax.set_xlim(
        (
            -item_width * 0.5 - margin,
            (item_width + margin) * (len(col_list) - 0.5) + margin * 0.5,
        )
    )

    if spine_mode == "tight":
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.plot(
            [ax.get_xlim()[0] + 0.000001] * 2,
            [ax.get_yticks()[0], ax.get_yticks()[-2]],
            linewidth=1.5,
            solid_capstyle="projecting",
            color="black",
        )
    elif spine_mode == "all":
        pass
    elif spine_mode == "open":
        ax.spines[["right", "top"]].set_visible(False)
    else:
        raise ValueError(
            "spine_mode should be 'open', 'tight', or 'all', not "
            + f"{spine_mode}.\n"
        )

    if rotate_labels == "auto":
        disp_label_lengths = np.array(
            [len(disp_names_dict[key]) for key in col_list]
        )
        rotate_labels = bool(
            np.max(disp_label_lengths[:-1] + disp_label_lengths[1:])
            / (item_width + margin)
            > 14
        )
    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for tick_text, key in zip(ax.get_xticklabels(), col_list):
        tick_text.set_color(tuple(col_colors_dict[key]))  # type: ignore

    # This reverses the order of the legend:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 1))

    ax = label_axes_etc(
        ax,
        title=title,
        y_label=y_label,
        disp_names=disp_names_dict,
    )

    if isinstance(save_path, (os.PathLike, str)):
        fig.savefig(
            save_path,  # type: ignore
            bbox_inches="tight",
            transparent=True,
            dpi=dpi,
        )

    if return_ax:
        return ax

    if show:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fig.show()

    elif not ax_in:
        plt.close(fig)


def legacy_column_plot(
    data_dict: dict[str, NDArray[np.float64]] | pd.DataFrame,  # {col_key:data}
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    y_label: str | None = None,
    title: str | None = None,
    ax_height: float = 2.5,
    item_width: float = 0.75,
    colors: dict[str, PossibleColor] | str | None = None,  # {col_key:color}
    disp_names: dict[str, str] | None = None,  # {col_key:display name}
    dpi: int | float = 600,
    v_range: tuple[float, float] | None | str = None,
    q_range: tuple[float, float] | None | str = "auto",
    min_data_density: float = 0.1,
    dot_size: float | str = "auto",
    color_col_labels: bool = True,
    color_points: bool | str = "auto",
    point_plot_type: str = "auto",  # Can be 'strip', 'swarm', or None.
    rotate_labels: bool | str = "auto",
    edge_color: Color | PossibleColor = (0, 0, 0, 1),
    edge_width: float = 0.5,
    vln_bw_adjust: float = 0.75,
    vln_grid_size: int = 400,
    shaded_range: _ShadedRangeType = None,
    shaded_range_color: Color | PossibleColor = (0.8, 0.1, 0, 0.3),
    spine_mode: str = "open",
    log_scale_y: bool = False,
    ax_in: None | Axes = None,
    show: bool = True,
    return_ax: bool = False,  # overrides show
    **kwargs: dict[str, str | float | None],
) -> None | Axes:
    """
    By default, this function makes a violin plot with some representation of
        the individual points. It could also be just swarm plots, box and
        whiskers, etc. though some functionality may not yet be implemented.

    Features to add:
    - hex color string support
    - improve interface for non-violin modes
    - horizontal mode

    Code style issues:
    - make new variables instead of changing the values of arguments
    """
    # Standardize argument types
    if isinstance(data_dict, pd.DataFrame):
        data_dict = {key: np.array(data_dict[key]) for key in data_dict.columns}
    else:  # Just in case the data are not already in numpy arrays.
        data_dict = {
            str(key): np.array(data_dict[str(key)]).flatten()
            for key in data_dict
        }

    if colors == "tab20":
        colors_dict = {
            str(key): tab20_colors[i] for i, key in enumerate(data_dict)
        }  # default is like Matplotlib

        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore

    elif (colors is None) or (colors is False) or (colors == "nb50"):
        colors_dict = {
            str(key): nb50_colors[i] for i, key in enumerate(data_dict)
        }

        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore

    elif isinstance(colors, dict):
        colors_dict = {}
        default_colors = nb50_colors
        default_colors_index = 0
        for key in data_dict:
            if key in colors:
                colors_dict[key] = tuple(colors[key])
            else:  # pass through on miss
                colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1

    else:
        raise ValueError(f"recieved bad kwarg colors={colors}")

    if (disp_names is None) or (disp_names is False):
        disp_names_dict = PassthroughDict(
            {str(key): str(key) for key in data_dict}
        )

    elif isinstance(disp_names, dict):
        disp_names_dict = PassthroughDict({})
        for key in data_dict:
            if key in disp_names:
                disp_names_dict[key] = disp_names[key]
            else:  # pass through on miss
                disp_names_dict[key] = key

    if (v_range == "auto") or (q_range == "auto"):
        n_bins: int = (
            100  # number of histogram bins
            #     (Used to check potential clipping and reduce plot range.)
        )
        clipping_aversion_factor: float = (
            3  # effectively increase top and bottom bin counts by this factor.
        )

        quantile_args = np.arange(1, n_bins + 1) / n_bins
        concatenated_data = np.hstack(
            [data_dict[str(key)] for key in data_dict]
        )
        quantile_values = np.quantile(concatenated_data, quantile_args)
        v_min = np.min([np.min(data_dict[key]) for key in data_dict])
        v_max = np.max([np.max(data_dict[key]) for key in data_dict])
        range_fractions = np.diff((quantile_values - v_min) / (v_max - v_min))
        range_fractions[0] *= clipping_aversion_factor
        range_fractions[-1] *= clipping_aversion_factor

        q_range = (
            quantile_args[
                np.min(
                    np.arange(len(range_fractions))
                    * (n_bins * range_fractions > min_data_density)
                )
            ],
            quantile_args[
                np.max(
                    np.arange(len(range_fractions))
                    * (n_bins * range_fractions > min_data_density)
                )
                + 1
            ],
        )

        v_range = None

    if (v_range is None) or (
        v_range is False
    ):  # v_range will override q_range if both are provided
        if (q_range is None) or (q_range is False):
            q_range = (0.0, 1.0)

        v_range = (
            float(
                np.min(
                    [
                        np.quantile(data_dict[key], float(q_range[0]))
                        for key in data_dict
                    ]
                )
            ),
            float(
                np.max(
                    [
                        np.quantile(data_dict[key], float(q_range[1]))
                        for key in data_dict
                    ]
                )
            ),
        )

    if dot_size == "auto":
        dot_size = np.clip(
            20
            / np.sqrt(
                np.max(
                    [
                        np.prod(np.array(data_dict[key].shape))
                        for key in data_dict
                    ]
                )
            ),
            0.1,
            20.0,
        )

    # Sets of kwargs to pass to functions inside
    sns_violin_kwargs = {"alpha", "inner", "density_norm"}

    if "inner" in kwargs:
        if not kwargs["inner"] in {"box", "quart", "point", "stick"}:
            inner = kwargs["inner"]
            kwargs["inner"] = None  # type: ignore
        else:
            inner = "pass"
    else:  # this leads to default behavior
        inner = "auto"
        kwargs["inner"] = None  # type: ignore

    # Derive substantially expanded range to clip data for violin
    #   and slightly expanded range for plot range.
    v_middle = np.mean((float(v_range[0]), float(v_range[1])))
    clip_max = v_middle + (float(v_range[1]) - v_middle) * 1.5
    clip_min = v_middle + (float(v_range[0]) - v_middle) * 1.5
    y_max = float(
        v_middle + (float(v_range[1]) - v_middle) * 1.0
    )  # (1 + np.sqrt(dot_size)*np.diff(v_range)/500)
    y_min = float(
        v_middle + (float(v_range[0]) - v_middle) * 1.0
    )  # (1 + np.sqrt(dot_size)*np.diff(v_range)/500)

    # Make matplotlib figure around axes of specified size.
    pad_factor: float = 1.25  # demoted from kwargs
    fig_size = (
        len(data_dict) * item_width * pad_factor,
        ax_height * pad_factor,
    )
    ax_position = (
        0.5 / pad_factor,
        0.5 / pad_factor,
        1 - 0.5 / pad_factor,
        1 - 0.5 / pad_factor,
    )
    if ax_in is None:
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes(ax_position)
    elif isinstance(ax_in, Axes):
        ax = ax_in
        fig = ax.get_figure()
        if not isinstance(fig, Figure):
            raise ValueError(
                f"Failed to get_figure from provided Axes ax_in={ax_in}"
            )
        fig.set_size_inches(*fig_size)
        ax.set_position(ax_position)
    else:
        raise TypeError(
            f"ax_in must be None or matplotlib.axes._axes.Axes, not {ax_in}."
        )

    # Make a version of the input data that has the final display names as keys
    #   and also clips data, if necessary.
    renamed_data_dict = {
        disp_names_dict[key]: np.clip(data_dict[key], clip_min, clip_max)
        for key in data_dict
    }
    clipped_data = {
        disp_names_dict[key]: np.clip(
            data_dict[key][(data_dict[key] < y_min) | (data_dict[key] > y_max)],
            v_range[0],
            v_range[1],
        )
        for key in data_dict
    }
    if point_plot_type in {"auto", "Auto"}:
        if (
            np.max(
                [
                    np.prod(renamed_data_dict[key].shape)
                    for key in renamed_data_dict
                ]
            )
            < 500
        ):
            point_plot_type = "swarm"
        else:
            point_plot_type = "strip"

    if color_points == "auto":
        color_points = point_plot_type == "swarm"

    if point_plot_type in {"strip", "Strip"}:
        if color_points:
            sns.stripplot(
                renamed_data_dict,
                ax=ax,
                size=float(dot_size),
                palette={
                    disp_names_dict[key]: colors_dict[key]  # type: ignore
                    for key in data_dict
                },
                jitter=0.45,
                zorder=0,
            )
        else:
            sns.stripplot(
                renamed_data_dict,
                ax=ax,
                size=float(dot_size),
                color="black",
                jitter=0.45,
                zorder=0,
            )
        sns.stripplot(
            clipped_data,
            ax=ax,
            size=float(dot_size),
            color="red",
            jitter=0.45,
            zorder=5,
        )  # clipped points

    elif point_plot_type in {"swarm", "Swarm"}:
        if color_points:
            sns.swarmplot(
                renamed_data_dict,
                ax=ax,
                size=float(dot_size),
                zorder=2,
                palette={
                    disp_names_dict[key]: colors_dict[key]  # type: ignore
                    for key in data_dict
                },
            )
        else:
            sns.swarmplot(
                renamed_data_dict,
                ax=ax,
                size=float(dot_size),
                color="black",
                zorder=0,
            )
        sns.swarmplot(
            clipped_data, ax=ax, size=float(dot_size), color="red", zorder=5
        )  # clipped points
    elif point_plot_type in {None, False}:
        pass
    else:
        raise ValueError(
            "point_plot_type should be 'strip', 'swarm', or None,"
            + f" not {point_plot_type}."
        )

    sns.violinplot(
        renamed_data_dict,
        ax=ax,
        saturation=1,
        zorder=1,
        common_norm=True,
        linewidth=edge_width,
        edgecolor=edge_color,
        bw_adjust=vln_bw_adjust,
        gridsize=vln_grid_size,
        palette={
            disp_names_dict[key]: colors_dict[key]  # type: ignore
            for key in data_dict
        },
        **{
            kwarg_key: kwargs[kwarg_key]
            for kwarg_key in kwargs
            if kwarg_key in sns_violin_kwargs
        },  # type: ignore
    )

    ax.set_ylim((y_min, y_max))

    if rotate_labels == "auto":
        disp_label_lengths = np.array(
            [len(disp_names_dict[key]) for key in data_dict]
        )
        rotate_labels = bool(
            np.max(disp_label_lengths[:-1] + disp_label_lengths[1:])
            / item_width
            > 14
        )
    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if color_col_labels:
        for tick_text, key in zip(ax.get_xticklabels(), colors_dict):
            tick_text.set_color(tuple(colors_dict[key]))  # type: ignore

    xlim = (  # This will break if plotting only 1 column...
        ax.get_xticks()[0] - 0.6 * (ax.get_xticks()[1] - ax.get_xticks()[0]),
        ax.get_xticks()[-1] + 0.6 * (ax.get_xticks()[-1] - ax.get_xticks()[-2]),
    )
    ax.set_xlim(xlim)

    if spine_mode == "tight":
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.plot(
            [ax.get_xlim()[0] + 0.000001] * 2,
            [ax.get_yticks()[0], ax.get_yticks()[-2]],
            linewidth=1.5,
            solid_capstyle="projecting",
            color="black",
        )
    elif spine_mode == "all":
        pass
    elif spine_mode == "open":
        ax.spines[["right", "top"]].set_visible(False)
    else:
        raise ValueError(
            "spine_mode should be 'open', 'tight', or 'all', not "
            + f"{spine_mode}.\n"
        )

    x_ticks = np.arange(len(data_dict), dtype=np.float64)  # default
    x_ticks[: len(ax.get_xticks())] = np.array(ax.get_xticks())

    if shaded_range is not None:
        if type(shaded_range) in {dict, tuple}:
            shaded_range = [shaded_range]  # type: ignore

        if isinstance(shaded_range, list):
            if (len(shaded_range) == 2) and (
                type(shaded_range[0]) in {int, float}
            ):
                # When using this function from R using reticulate,
                #   list may show up that should have been a tuple.
                shaded_range = [tuple(shaded_range)]  # type: ignore

            for particular_range in shaded_range:  # type: ignore
                if isinstance(particular_range, dict):
                    col_edges = (
                        [-0.5]
                        + list((x_ticks[:-1] + x_ticks[1:]) / 2)
                        + [0.5 + len(x_ticks)]
                    )
                    for i_col, col_key in enumerate(data_dict):
                        ax.fill_between(
                            [col_edges[i_col], col_edges[i_col + 1]],
                            [particular_range[col_key][0]] * 2,
                            [particular_range[col_key][1]] * 2,
                            color=shaded_range_color,
                            zorder=6,
                            linewidth=0,
                            edgecolor=edge_color,
                        )

                elif isinstance(particular_range, tuple):
                    ax.fill_between(
                        [
                            x_ticks[0] - 1 / 2,
                            x_ticks[-1] + 1 / 2,
                        ],
                        [particular_range[0]] * 2,
                        [particular_range[1]] * 2,
                        color=shaded_range_color,
                        zorder=6,
                        linewidth=0,
                        edgecolor=edge_color,
                    )
                else:
                    raise TypeError(
                        "All shaded range must be tuple or dict, "
                        + "not {shaded_range}.\n"
                    )

    if inner in {
        "line_box",
        "auto",
        None,
    }:  # custom box plot made from individual lines
        box_width = 0.25 * item_width
        linewidth = 1
        x_nudge = 0  # unused; would nudge the box plots to the side
        for i_col, key in enumerate(data_dict):
            quartiles = np.quantile(data_dict[key], [0.0, 0.25, 0.5, 0.75, 1.0])
            ax.plot(  # lower quartile
                [
                    x_nudge + x_ticks[i_col] - 0.5 * box_width,
                    x_nudge + x_ticks[i_col] + 0.5 * box_width,
                ],
                [quartiles[1]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # median
                [
                    x_nudge + x_ticks[i_col] - 0.5 * box_width,
                    x_nudge + x_ticks[i_col] + 0.5 * box_width,
                ],
                [quartiles[2]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # upper quartile
                [
                    x_nudge + x_ticks[i_col] - 0.5 * box_width,
                    x_nudge + x_ticks[i_col] + 0.5 * box_width,
                ],
                [quartiles[3]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # left side
                [x_nudge + x_ticks[i_col] - 0.5 * box_width] * 2,
                [quartiles[1], quartiles[3]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # right side
                [x_nudge + x_ticks[i_col] + 0.5 * box_width] * 2,
                [quartiles[1], quartiles[3]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # upper whisker
                [x_nudge + x_ticks[i_col]] * 2,
                [quartiles[1], quartiles[0]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # lower whisker
                [x_nudge + x_ticks[i_col]] * 2,
                [quartiles[3], quartiles[4]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )

    if isinstance(y_label, str):
        ax.set_ylabel(y_label)

    if isinstance(title, str):
        ax.set_title(title)

    ax = label_axes_etc(
        ax,
        log_scale_y=log_scale_y,
        title=title,
        y_label=y_label,
        disp_names=disp_names_dict,
    )

    if isinstance(save_path, (os.PathLike, str)):
        fig.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)

    if return_ax:
        return ax

    if show:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fig.show()


def legacy_scatter_plot(
    data: pd.DataFrame | dict[str, Arrayable] | Arrayable,  # type: ignore
    *y_c: tuple[Arrayable, Arrayable],  # type: ignore
    mode: str = "categorical",  # Could be "color_map"
    pull_labels_from_data: bool = False,
    color_palette: dict[str, Color] | str = "nb50",
    size: float | Arrayable | str = "auto",  # type: ignore
    disp_names: dict[str | int, str] | None = None,
    mix_mode: str = "z_shuffle",  # 'z_shuffle', 'ordered'
    axes_dimensions: tuple[float, float] | None = None,
    ax_in: Axes | None = None,
    dpi: int = 600,
    x_label: str | None = None,
    y_label: str | None = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    title: str | None = None,
    title_color: Color | str = "black",
    color_legend_title: str | None = "",
    size_legend_title: str | None = "",
    show_grid: bool | str = False,  # can be "major", "minor", or "both"
    include_cat_legend: bool = True,
    include_size_legend: bool = True,
    size_to_area_func: Callable[[NDArray], NDArray] | None = None,
    cmap: str | Colormap = "YlOrBr",
    cmap_norm: Callable | None = None,
    colorbar_label: str | None = None,
    show_colorbar: bool = True,
    autoscale_size: bool = True,
    hide_spines: bool | list[str] = False,
    aspect_ratio: float | None = None,
    axis_off: bool = False,
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    show: bool = True,
    return_ax: bool = False,  # overrides show
    **kwargs,
) -> None | Axes:
    """
    This function makes a scatter plot where the points each point is colored by
    one of a dicrete set of categories (e.g. clusters, conditions).

    feature ideas:
     - allow the use of data frame column names as axis labels.
    """

    if not axes_dimensions:
        if x_lim and y_lim and aspect_ratio:
            if isinstance(aspect_ratio, float):
                ax_dims = (
                    2.5 * (y_lim[1] - y_lim[0]) / aspect_ratio,
                    2.5,
                )  # check
            elif aspect_ratio == "equal":
                ax_dims = (2.5 * (y_lim[1] - y_lim[0]), 2.5)  # check
            else:
                ax_dims = (2.5, 2.5)
        else:
            ax_dims = (2.5, 2.5)
    elif x_lim and y_lim and aspect_ratio:
        raise TypeError(
            "Specifying ax_dims, x_lim, y_lim, and aspect_ratio overdefines ax."
        )
    else:
        ax_dims = axes_dimensions

    if not mode in {"categorical", "color_map"}:
        raise ValueError(
            f"mode must be 'categorical' or 'color_map', not {mode}.\n"
        )
    if len(y_c) in {
        1,
        2,
    }:  # recieved >1 args -> interpret as x, y, and maybe c.
        if isinstance(data, dict):
            raise TypeError(
                "If multiple args are provided, the first will be interpreted"
                + "as X-values and the second as Y-values. Try providing first "
                + "arg of type NDArray, list, or Series, not dict.\n"
            )
        x_raw = np.array(standardize_arrayable(data))
        y_raw = np.array(standardize_arrayable(y_c[0]))  # type: ignore
        if (mode == "categorical") and (len(y_c) == 2):
            c_raw = list(
                standardize_arrayable(y_c[1], dtype="list")  # type: ignore
            )
        elif mode == "categorical":
            raise TypeError(
                "for mode='categorical', number of args "
                + f"must be 1 or 3 not {len(y_c) + 1}.\n"
            )
        elif mode == "color_map":
            if isinstance(y_c[1], (list, np.ndarray, Series, pd.DataFrame)):
                c_raw = np.array(
                    standardize_arrayable(
                        y_c[1],  # type: ignore
                    )
                )
            else:
                raise TypeError()
        else:
            raise ValueError()

    elif len(y_c) > 2:
        raise TypeError(
            f"Number of args must be 1, 2, or 3 not {len(y_c) + 1}.\n"
        )

    else:  # if (len(y_c) == 0), data arg must contain x, y, and possibly c.
        if isinstance(data, dict):
            if ("x" in data) and ("y" in data) and ("c" in data):
                x_raw = np.array(standardize_arrayable(data["x"]))
                y_raw = np.array(standardize_arrayable(data["y"]))
                if mode == "categorical":
                    c_raw = list(standardize_arrayable(data["c"], dtype="list"))
                elif mode == "color_map":
                    c_raw = np.array(
                        standardize_arrayable(data["c"], dtype=np.float64)
                    )
            elif len(data) > 2:
                x_raw = np.array(standardize_arrayable(data[list(data)[0]]))
                y_raw = np.array(standardize_arrayable(data[list(data)[1]]))
                if mode == "categorical":
                    c_raw = list(
                        standardize_arrayable(data[list(data)[2]], dtype="list")
                    )
                elif mode == "color_map":
                    c_raw = np.array(
                        standardize_arrayable(
                            data[list(data)[2]], dtype=np.float64
                        )
                    )
            else:
                raise TypeError(
                    "data of type dict must have keys 'x', and 'y'.\n"
                )

        elif isinstance(data, pd.DataFrame):
            data_dropna = data.dropna(axis="rows")  # type: ignore
            if (
                ("x" in data.columns)
                and ("y" in data.columns)
                and ("c" in data.columns)
            ):
                x_raw = data_dropna["x"]
                y_raw = data_dropna["y"]
                if mode == "categorical":
                    c_raw = list(data_dropna["c"])
                elif mode == "color_map":
                    c_raw = standardize_arrayable(
                        data_dropna["c"], dtype=np.float64
                    )
            elif len(data.columns) >= 3:
                x_raw = data_dropna[data.columns[0]]
                y_raw = data_dropna[data.columns[1]]
                if mode == "categorical":
                    c_raw = list(data_dropna[data.columns[2]])
                elif mode == "color_map":
                    c_raw = standardize_arrayable(
                        data_dropna[data.columns[2]], dtype=np.float64
                    )
            else:
                raise ValueError(
                    "data of type DataFrame must have columns"
                    + " 'x' and 'y', and 'c' for mode='categorical'.\n"
                )
        else:
            raise TypeError(
                f"Try providing dict or DataFrame with columns"
                + " 'x', 'y', and 'c'.\n"
            )

    # get s_raw, a list of the sizes of each point on the plot.
    if isinstance(size, (float, int)):
        s_raw = size * np.ones(len(x_raw))
    elif isinstance(size, (list, np.ndarray, Series, pd.DataFrame)):
        s_raw = standardize_arrayable(size)
        if not len(s_raw) == len(x_raw):
            raise ValueError(
                "'size' of type list, np.ndarray, Series, or pd.DataFrame "
                + "must the same length as X vector taken form first arg.\n"
                + f"len(X) = {len(x_raw)},\nlen(size) = {len(s_raw)}.\n"
            )
    elif size == "auto":
        uniform_s = np.ones(len(x_raw)) * 100 / np.sqrt(10 + len(x_raw))
        if isinstance(data, (dict, pd.DataFrame)) and (len(y_c) == 0):
            if "s" in data:
                s_raw = np.array(standardize_arrayable(data["s"]))
            else:
                s_raw = uniform_s
        else:
            s_raw = uniform_s
    else:
        raise TypeError(
            f"bad kwarg size = {size}; must be list, float, or 'auto'.\n"
        )

    if size_to_area_func is None:
        size_func = lambda x: x
    elif callable(size_to_area_func):
        size_func = size_to_area_func
    else:
        raise TypeError(
            "size_to_area_func should be callable or None, "
            + "Not {size_to_area_func}.\n"
        )

    if not len(x_raw) == len(y_raw) == len(s_raw):
        raise ValueError(
            f"incompatible lengths:\n"
            + f"    len(x)={len(x_raw)}\n"
            + f"    len(y)={len(y_raw)}\n"
            + f"    len(s)={len(s_raw)}\n"
        )

    if mode == "categorical":  # set up colors for "categorical" mode.
        if not len(c_raw) == len(x_raw):  # type: ignore
            raise ValueError(
                f"incompatible lengths:\n"
                + f"    len(x)={len(x_raw)}\n"
                + f"    len(y)={len(y_raw)}\n"
                + f"    len(c)={len(c_raw)}\n"  # type: ignore
            )

        cat_set = set(c_raw)  # type: ignore

        if color_palette == "tab20":
            colors_dict = {
                i: color
                for i, color in enumerate(tab20_colors)  # like Matplotlib
            }
            n_colors = 20

        elif (
            (color_palette == "nb50")
            or (color_palette is False)
            or (color_palette is None)
        ):
            colors_dict = {
                i: color
                for i, color in enumerate(nb50_colors)  # like Matplotlib
            }
            n_colors = 50

        elif isinstance(color_palette, dict):
            colors_dict = color_palette
            default_colors = {
                i: color
                for i, color in enumerate(nb50_colors)  # like Matplotlib
            }
            default_colors_index = 0
            for key in cat_set:  # type: ignore
                if not key in color_palette:
                    colors_dict[key] = default_colors[
                        default_colors_index % len(default_colors)
                    ]
                    default_colors_index += 1
            n_colors = len(colors_dict)

        else:
            raise TypeError(f"bad kwarg color_palette = {color_palette}")

        # verify that needed variables have actually been defined.
        if ("c_raw" not in locals()) and (mode == "categorical"):
            raise UnboundLocalError("unbound variable c_raw (unexpected)")

        if "colors_dict" not in locals():
            raise UnboundLocalError("unbound variable colors_dict (unexpected)")

        if "n_colors" not in locals():
            raise UnboundLocalError("unbound variable n_colors (unexpected)")

        if not all([c in colors_dict for c in c_raw]):  # type: ignore
            # more work will be requred to map c to color_palette
            if all(isinstance(key, int) for key in colors_dict.keys()):
                try:
                    # if all elements of c can be converted to integers,
                    # this will run without an error:
                    int_c_raw = {int(cat) for cat in c_raw}  # type: ignore
                except ValueError:
                    # In this case, categories will be arbitrarily mapped to
                    # colors. This could lead to inconsistent color coding
                    # between plots, but might be intentional.
                    cat_to_color_key = {
                        cat: i % n_colors for i, cat in enumerate(cat_set)
                    }
                    if not all(
                        [(k in colors_dict) for k in cat_to_color_key.values()]
                    ):
                        raise ValueError(
                            "Color mapping error. kwarg colors ="
                            + f" {color_palette} may be a dict with non-"
                            + "sequential integer keys that do not exactly "
                            + f"match the values of 'c', {cat_set}."
                        )
                except Exception as e:
                    raise ValueError("Unexpected error in color mapping.")
                else:
                    cat_to_color_key = {
                        cat: int(cat) % n_colors for cat in set(int_c_raw)
                    }

        else:
            cat_to_color_key = {cat: cat for cat in colors_dict}

    else:
        cat_set = {"mittens", "leo"}  # (joke and Pylance appeasement)

    if disp_names:
        if isinstance(disp_names, dict):
            cat_to_disp_name = PassthroughDict({})
            for key in cat_set:
                if key in disp_names:
                    cat_to_disp_name[key] = disp_names[key]
                else:  # pass through on miss
                    cat_to_disp_name[key] = key
        else:
            raise TypeError(
                "If provided, disp names should be of type dict, "
                + f"not {type(disp_names)}"
            )

        label_to_disp_name = PassthroughDict({})
        if not y_label is None:
            if y_label in disp_names:
                label_to_disp_name[y_label] = disp_names[y_label]
        if not x_label is None:
            if x_label in disp_names:
                label_to_disp_name[x_label] = disp_names[x_label]
        if not title is None:
            if title in disp_names:
                label_to_disp_name[title] = disp_names[title]
    else:
        if mode == "categorical":
            cat_to_disp_name = PassthroughDict({c: str(c) for c in cat_set})

        label_to_disp_name = {}
        if not y_label is None:
            label_to_disp_name[y_label] = y_label
        if not x_label is None:
            label_to_disp_name[x_label] = x_label
        if not title is None:
            label_to_disp_name[title] = title

    if mix_mode == "z_shuffle":
        point_order = np.random.permutation(len(x_raw))
    elif mix_mode == "ordered":
        point_order = np.arange(len(x_raw))
    else:
        raise ValueError(
            f"mix_mode {mix_mode} not recognized; try 'z_shuffle' or 'ordered'"
        )

    # prepare arguments for ax.scatter()
    x_sorted = x_raw[point_order]
    y_sorted = y_raw[point_order]
    if mode == "categorical":
        color_list = [
            colors_dict[cat_to_color_key[c_raw[i]]]  # type: ignore
            for i in point_order
        ]
    elif mode == "color_map":
        c_sorted = c_raw[point_order]

    s_array = size_func(s_raw[point_order])

    if autoscale_size and isinstance(
        size, (np.ndarray, list, pd.DataFrame, Series)
    ):
        s_scale_factor = np.minimum(
            100 / np.max(s_array), 30 / np.median(s_array)
        )
    else:
        s_scale_factor = 1

    s_array = float(s_scale_factor) * s_array

    # Make matplotlib figure around axes of specified size.
    pad_factor = 1.25
    ax_position = (
        0.5 / pad_factor,
        0.5 / pad_factor,
        1 - 0.5 / pad_factor,
        1 - 0.5 / pad_factor,
    )
    fig_size = (ax_dims[0] * pad_factor, ax_dims[1] * pad_factor)

    if ax_in is None:
        fig = plt.figure(figsize=fig_size, dpi=dpi)
        ax = fig.add_axes(ax_position)
    elif isinstance(ax_in, Axes):
        ax = ax_in
        fig = ax.get_figure()
        if not isinstance(fig, Figure):
            raise ValueError(
                f"Failed to get_figure from provided Axes ax_in={ax_in}"
            )
        fig.set_size_inches(*fig_size)  # type: ignore
        ax.set_position(ax_position)
    else:
        raise TypeError(
            f"ax_in must be None or matplotlib.axes._axes.Axes, not {ax_in}."
        )

    if not show_grid is False:
        ax.set_axisbelow(True)
        if show_grid in {"major", "minor", "both"}:
            ax.grid(
                True, which=show_grid, zorder=-1, linewidth=0.5  # type: ignore
            )
        else:
            ax.grid(True, zorder=-1, linewidth=0.5)

    if pull_labels_from_data:
        label_kwargs = labels_from_data(
            data,
            *y_c,
            (
                size
                if len(y_c) == 2  # (this is to make size sure size is args[3])
                else (
                    (None, size)
                    if len(y_c) == 1
                    else (None, None, size) if len(y_c) == 0 else None
                )
            ),
            x_label=x_label,
            y_label=y_label,
            color_data_label=(
                color_legend_title if mode == "categorical" else colorbar_label
            ),
            size_legend_title=size_legend_title,
        )
    else:
        label_kwargs = {
            "x_label": x_label,
            "y_label": y_label,
            "color_data_label": (
                color_legend_title
                if (mode == "categorical")
                else colorbar_label
            ),
            "size_legend_title": size_legend_title,
        }

    scatter_kwargs = {"marker", "alpha", "edgecolors"}

    if mix_mode in {"z_shuffle", "ordered"}:
        if mode == "categorical":
            ax.scatter(
                x_sorted,
                y_sorted,
                c=color_list,  # type: ignore
                s=size_func(s_raw[point_order]) * s_scale_factor,
                linewidths=0,
                **{
                    key: val  # type: ignore
                    for key, val in kwargs.items()
                    if key in scatter_kwargs
                },
            )
        elif mode == "color_map":
            if isinstance(cmap_norm, (Normalize, LogNorm)):
                norm = cmap_norm
            elif cmap_norm == "log":
                norm = LogNorm()
            elif isinstance(cmap_norm, (tuple, list)):
                if len(cmap_norm) == 3:
                    if cmap_norm[2] == "log":
                        norm = LogNorm(vmin=cmap_norm[0], vmax=cmap_norm[1])
                    elif cmap_norm[2] == "linear":
                        norm = Normalize(vmin=cmap_norm[0], vmax=cmap_norm[1])
                    else:
                        raise ValueError(
                            "last element of cmap_norm of length 3 should be "
                            + f"either 'log' or 'linear' not {cmap_norm[2]}.\n"
                        )
                elif len(cmap_norm) == 2:
                    norm = Normalize(vmin=cmap_norm[0], vmax=cmap_norm[1])
            elif (cmap_norm == "linear") or (cmap_norm is None):
                norm = Normalize()

            scat = ax.scatter(
                x_sorted,
                y_sorted,
                c=c_sorted,
                s=size_func(s_raw[point_order]) * s_scale_factor,
                cmap=cmap,
                norm=norm,
                linewidths=0,
                **{
                    key: val  # type: ignore
                    for key, val in kwargs.items()
                    if key in scatter_kwargs
                },
            )

            if show_colorbar:
                cbar_ax_gap_width = 0.04
                cax = fig.add_axes(
                    (
                        ax_position[0] + ax_position[2] + cbar_ax_gap_width,
                        ax_position[1],
                        cbar_ax_gap_width,
                        ax_position[3],
                    )
                )

                fig.colorbar(
                    scat, cax=cax, label=label_kwargs["color_data_label"]
                )

            ax.reset_position()

    else:  # If a mode was added to the previous mix_mode check but not here.
        raise ValueError(
            f"mix_mode {mix_mode} not recognized. Try 'z_shuffle' or 'ordered'"
        )

    ax = label_axes_etc(
        ax,
        ax_dims=ax_dims,
        x_lim=x_lim,
        y_lim=y_lim,
        log_scale_x=log_scale_x,
        log_scale_y=log_scale_y,
        disp_names=PassthroughDict(label_to_disp_name),
        title_color=title_color,
        aspect_ratio=aspect_ratio,
        hide_spines=hide_spines,
        axis_off=axis_off,
        title=title,
        **label_kwargs,
    )

    if include_cat_legend and (mode == "categorical"):
        key_to_cat = PassthroughDict(cat_to_color_key).inverse  # type: ignore
        handles, labels = surrogate_color_legend_info(
            {
                key_to_cat[key]: colors_dict[key]  # type: ignore
                for key in colors_dict  # type: ignore
                if key in cat_to_color_key.values()  # type: ignore
                and key_to_cat[key] in cat_set
            },  # type: ignore
            cat_to_disp_name,  # type: ignore
            **{
                key: val  # type: ignore
                for key, val in kwargs.items()
                if key in scatter_kwargs ^ {"alpha"}
            },
        )

        color_ghost_ax = ax.twinx()
        color_ghost_ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1 + 0.2 / ax_dims[0], 0.5),
            markerscale=30 / (20 + len(handles)),
            fancybox=True,
            title=label_kwargs["color_data_label"],
            fontsize=200 / (20 + len(labels)),
            # title_fontsize=250 / (20 + len(labels)), # good idea?
        )
        color_ghost_ax.get_yaxis().set_visible(False)
        color_ghost_ax.spines[["left", "right", "top", "bottom"]].set_visible(
            False
        )

    if include_size_legend and isinstance(
        size, (np.ndarray, list, pd.DataFrame, Series)
    ):
        # if applicable, a marker size legend will be added to the top of the
        # plot above title. The assumption is that the image will ultimately be
        # split to independedntly move the legend and plot in a figure layout
        # anyway. A possible future feature could be to put it somewhere better.
        handles, labels = surrogate_size_legend_info_automatic(
            s_raw[point_order],
            size_to_area_func=lambda x: size_func(x) * s_scale_factor,
            **{
                key: val  # type: ignore
                for key, val in kwargs.items()
                if key in scatter_kwargs ^ {"alpha"}
            },
        )
        size_ghost_ax = ax.twinx()
        size_ghost_ax.legend(
            handles,
            labels,
            loc="center right",
            bbox_to_anchor=(-1 / ax_dims[0], 0.5),
            markerscale=1,
            fancybox=True,
            title=label_kwargs["size_legend_title"],
            fontsize=160 / (20 + len(labels)),
            # title_fontsize=200 / (20 + len(labels)),
        )
        size_ghost_ax.get_yaxis().set_visible(False)
        size_ghost_ax.spines[["left", "right", "top", "bottom"]].set_visible(
            False
        )

    if isinstance(save_path, (str, os.PathLike)):
        fig.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)

    if return_ax:
        return ax

    if show:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            fig.show()
    else:
        plt.close(fig)
