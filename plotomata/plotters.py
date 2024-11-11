"""
This module contains the actual plotting functions for plotomata package.
"""

import os
import sys
from typing import TypeAlias
import importlib
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd

# If this doesn't make sense, just pretend it says "from .colors import ..."
sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
import colors
from colors import tab20_colors, nb50_colors, Color, ListColor

importlib.reload(colors)


_ShadedRangeType: TypeAlias = (
    None
    | list[tuple[float, float] | dict[str, tuple[float, float]]]
    | tuple[float, float]
    | dict[str, tuple[float, float]]
)


def bar_plot(
    data: pd.DataFrame,
    colors: None | str | dict[str, Color | ListColor] = None,
    col_colors: None | str | dict[str, Color | ListColor] = None,
    mode: str = "stacked",
    disp_names: dict[str, str] | None = None,
    column_order: list[int] | NDArray | None = None,
    row_order: list[int] | NDArray | None = None,
    item_width: float = 0.5,
    margin: float = 0.1,
    ax_height: float = 2.5,
    dpi: int = 600,
    edge_color: Color | ListColor = (0, 0, 0, 1),
    edge_width: float = 0.5,
    rotate_labels: bool | str = "auto",
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    y_label: str | None = None,
    title: str | None = None,
    spine_mode: str = "tight",
    show: bool = True,
    ax_in: None | Axes = None,
    **kwargs: dict[str, str | float | None],
) -> None:
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
        disp_names_dict = disp_names
    elif disp_names is None:
        disp_names_dict = {
            key: key for key in list(data_df.index) + list(data_df.columns)
        }
    else:
        raise TypeError("disp_names should be dict[str: str] type.")

    if column_order is None:
        col_list = [str(key) for key in data_df.columns]
    else:
        col_list = [str(data_df.columns[i_key]) for i_key in column_order]

    if row_order is None:
        row_list = [str(key) for key in data_df.index]
    else:
        row_list = [str(data_df.index[i_row]) for i_row in row_order]

    if colors == "tab20":
        colors_dict = {
            str(key): tab20_colors[i] for i, key in enumerate(row_list)
        }  # default is like Matplotlib

        if not "alpha" in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore
    elif (colors is None) or (colors == "nb50"):
        colors_dict = {
            str(key): nb50_colors[i]
            for i, key in enumerate(row_list)  # black lint
        }
    elif isinstance(colors, dict):
        colors_dict = colors
    else:
        raise ValueError(f"colors={colors} not supported.")

    if col_colors == "tab20":
        col_colors_dict = {
            str(key): tab20_colors[i] for i, key in enumerate(col_list)
        }  # default is like Matplotlib
    elif col_colors == "nb50":
        col_colors_dict = {
            str(key): nb50_colors[i]
            for i, key in enumerate(col_list)  # black lint
        }
    elif isinstance(col_colors, dict):
        col_colors_dict = col_colors
    elif col_colors is None:
        col_colors_dict = {
            str(key): (0, 0, 0, 1)
            for i, key in enumerate(col_list)  # black lint
        }
    elif col_colors == "from_colors":
        col_colors_dict = colors_dict
    else:
        raise ValueError(f"col_colors={col_colors} not supported.")

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
        if isinstance(fig, Figure):
            raise ValueError(  # black lint
                f"Failed to get_figure from provided Axes ax_in={ax_in}"
            )
        fig.set_size_inches(*fig_size)  # type: ignore
        ax.set_position(ax_position)
    else:
        raise TypeError(
            f"ax_in must be None or matplotlib.axes._axes.Axes, not {ax_in}."
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
                    kwarg_key: kwargs[kwarg_key]
                    for kwarg_key in kwargs
                    if kwarg_key in plt_bar_kwargs
                },  # type: ignore
            )
            bottoms = [
                height + bottom  # black lint
                for height, bottom in zip(heights, bottoms)
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
        raise ValueError(f'mode={mode} not supported; try mode="stacked"')

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
    else:
        ax.spines[["right", "top", "left"]].set_visible(False)

    if rotate_labels == "auto":
        disp_label_lengths = np.array(
            [len(disp_names_dict[key]) for key in col_list]  # black lint
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
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.5))

    if not y_label is None:
        ax.set_ylabel(y_label)

    if not title is None:
        ax.set_title(title)

    if not save_path is None:
        fig.savefig(  # type: ignore
            save_path, bbox_inches="tight", transparent=True, dpi=dpi
        )

    if show:
        fig.show()  # type: ignore
    elif ax_in is None:
        plt.close(fig)


def column_plot(
    data_dict: dict[str, NDArray[np.float64]] | pd.DataFrame,  # {col_key:data}
    show: bool = True,
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    y_label: str | None = None,
    title: str | None = None,
    ax_height: float = 2.5,
    item_width: float = 0.75,
    colors: dict[str, Color | ListColor] | str | None = None,  # {col_key:color}
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
    edge_color: Color | ListColor = (0, 0, 0, 1),
    edge_width: float = 0.5,
    vln_bw_adjust: float = 0.75,
    vln_grid_size: int = 400,
    shaded_range: _ShadedRangeType = None,
    shaded_range_color: Color | ListColor = (0.8, 0.1, 0, 0.3),
    spine_mode: str = "tight",
    ax_in: None | Axes = None,
    **kwargs: dict[str, str | float | None],
) -> None:
    """
    By default, this function makes a violin plot with some representation of
        the individual points.

    Features to add:
    - hex color string support
    - improve interface for non-violin modes

    Code style issues:
    - make new variables instead of changing the values of arguments
    """

    # Standardize argument types
    if isinstance(data_dict, pd.DataFrame):
        data_dict = {key: np.array(data_dict[key]) for key in data_dict.columns}
    else:  # Just in case the data are not already in numpy arrays.
        data_dict = {
            str(key): np.array(data_dict[str(key)]).flatten()  # black lint
            for key in data_dict
        }

    if colors == "tab20":
        colors = {
            str(key): tab20_colors[i] for i, key in enumerate(data_dict)
        }  # default is like Matplotlib

        if not "alpha" in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore

    elif (colors is None) or (colors == "nb50"):
        colors = {str(key): nb50_colors[i] for i, key in enumerate(data_dict)}

        if not "alpha" in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore

    if disp_names == None:
        disp_names = {str(key): str(key) for key in data_dict}

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
            [data_dict[str(key)] for key in data_dict]  # black lint
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

    if v_range is None:  # v_range will override q_range if both are provided
        if q_range is None:
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
                        np.prod(np.array(data_dict[key].shape))  # black lint
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
    fig_size = (  # black lint
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
            raise ValueError(  # black lint
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
        disp_names[key]: np.clip(data_dict[key], clip_min, clip_max)
        for key in data_dict
    }
    clipped_data = {
        disp_names[key]: np.clip(
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
                    np.prod(renamed_data_dict[key].shape)  # black lint
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
                    disp_names[key]: colors[key]  # type: ignore
                    for key in data_dict  # black lint
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
                    disp_names[key]: colors[key]  # type: ignore
                    for key in data_dict  # black lint
                },
            )
        else:
            sns.swarmplot(
                renamed_data_dict,
                ax=ax,  # black lint
                size=float(dot_size),
                color="black",
                zorder=0,
            )
        sns.swarmplot(
            clipped_data, ax=ax, size=float(dot_size), color="red", zorder=5
        )  # clipped points
    elif point_plot_type in {None, "none", "None"}:
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
            disp_names[key]: colors[key] for key in data_dict  # type: ignore
        },  # black lint
        **{
            kwarg_key: kwargs[kwarg_key]
            for kwarg_key in kwargs
            if kwarg_key in sns_violin_kwargs
        },  # type: ignore
    )

    ax.set_ylim((y_min, y_max))

    if rotate_labels == "auto":
        disp_label_lengths = np.array(
            [len(disp_names[key]) for key in data_dict]  # black lint
        )
        rotate_labels = bool(
            np.max(
                disp_label_lengths[:-1] + disp_label_lengths[1:]
            )  # black lint
            / item_width
            > 14
        )
    if rotate_labels:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if color_col_labels:
        for tick_text, key in zip(ax.get_xticklabels(), colors):
            tick_text.set_color(tuple(colors[key]))  # type: ignore

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
    else:
        ax.spines[["right", "top", "left"]].set_visible(False)

    if not shaded_range is None:
        if type(shaded_range) in {dict, tuple}:
            shaded_range = [shaded_range]  # type: ignore

        if isinstance(shaded_range, list):
            if (len(shaded_range) == 2) and (  # black lint
                type(shaded_range[0]) in {int, float}
            ):
                # When using this function from R using reticulate,
                #   list may show up that should have been a tuple.
                shaded_range = [tuple(shaded_range)]  # type: ignore

            for particular_range in shaded_range:  # type: ignore
                if type(particular_range) is dict:

                    col_edges = (
                        [-0.5]
                        + list(
                            (
                                np.array(ax.get_xticks()[:-1])
                                + np.array(ax.get_xticks()[1:])
                            )
                            / 2
                        )
                        + [0.5 + len(ax.get_xticks())]
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

                else:
                    assert type(particular_range) == tuple
                    ax.fill_between(
                        [
                            ax.get_xticks()[0] - 1 / 2,  # black lint
                            ax.get_xticks()[-1] + 1 / 2,
                        ],
                        [particular_range[0]] * 2,
                        [particular_range[1]] * 2,
                        color=shaded_range_color,
                        zorder=6,
                        linewidth=0,
                        edgecolor=edge_color,
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
                    x_nudge + ax.get_xticks()[i_col] - 0.5 * box_width,
                    x_nudge + ax.get_xticks()[i_col] + 0.5 * box_width,
                ],
                [quartiles[1]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # median
                [
                    x_nudge + ax.get_xticks()[i_col] - 0.5 * box_width,
                    x_nudge + ax.get_xticks()[i_col] + 0.5 * box_width,
                ],
                [quartiles[2]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # upper quartile
                [
                    x_nudge + ax.get_xticks()[i_col] - 0.5 * box_width,
                    x_nudge + ax.get_xticks()[i_col] + 0.5 * box_width,
                ],
                [quartiles[3]] * 2,
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # left side
                [x_nudge + ax.get_xticks()[i_col] - 0.5 * box_width] * 2,
                [quartiles[1], quartiles[3]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # right side
                [x_nudge + ax.get_xticks()[i_col] + 0.5 * box_width] * 2,
                [quartiles[1], quartiles[3]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # upper whisker
                [x_nudge + ax.get_xticks()[i_col]] * 2,
                [quartiles[1], quartiles[0]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )
            ax.plot(  # lower whisker
                [x_nudge + ax.get_xticks()[i_col]] * 2,
                [quartiles[3], quartiles[4]],
                color="black",
                solid_capstyle="round",
                linewidth=linewidth,
                zorder=4,
            )

    if not y_label is None:
        ax.set_ylabel(y_label)

    if not title is None:
        ax.set_title(title)

    if not save_path == None:
        fig.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)

    if show:
        fig.show()
