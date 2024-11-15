"""
This module contains the actual plotting functions for plotomata package.
"""

try:
    from . import color_sets
    from .color_sets import tab20_colors, nb50_colors, Color, ListColor
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_sets
        from color_sets import tab20_colors, nb50_colors, Color, ListColor
    except ImportError as ie2:
        raise ie2 from ie
except Exception as e:
    raise e
else:
    import os

from typing import TypeAlias, Callable
import warnings
import importlib
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", module="matplotlib")
import seaborn as sns
import pandas as pd


importlib.reload(color_sets)


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
    spine_mode: str = "open", # could also be "tight" or "all"
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
        disp_names_dict = {}
        for key in list(data_df.index) + list(data_df.columns):
            if key in disp_names:
                disp_names_dict[key] = disp_names[key]
            else: # pass through on miss
                disp_names_dict[key] = key
    elif (disp_names is None) or (disp_names is False):
        disp_names_dict = {
            key: key for key in list(data_df.index) + list(data_df.columns)
        }
    else:
        raise TypeError(
            f"disp_names should be of type dict, not {type(disp_names)}."
        )

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

        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.5  # type: ignore
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
            else: # pass through on miss
                colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1
    else:
        raise ValueError(f"colors={colors} not supported.")

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
            str(key): (0, 0, 0, 1) for key in col_list # default: all black
        }
    elif col_colors == "from_colors":
        if (colors is None) or (colors == False):
            col_colors_dict = {
                str(key): (0, 0, 0, 1) for key in col_list # default: all black
            }
        else:
            col_colors_dict = {}
            for key in col_list:
                if key in colors_dict:
                    col_colors_dict[key] = colors_dict[key]
                else:
                    col_colors_dict[key] = (0, 0, 0, 1)

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
    elif spine_mode == "open":
        ax.spines[["right", "top"]].set_visible(False)
    else:
        raise ValueError(
            f"spine_mode should be 'open', 'tight', or 'all', not {spine_mode}"
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
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0.5))

    if y_label:
        ax.set_ylabel(y_label) # type: ignore

    if title:
        ax.set_title(title) # type: ignore

    if isinstance(save_path, (os.PathLike, str)):
        fig.savefig(
            save_path,  # type: ignore
            bbox_inches="tight", 
            transparent=True, 
            dpi=dpi
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
    spine_mode: str = "open",
    ax_in: None | Axes = None,
    **kwargs: dict[str, str | float | None],
) -> None:
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
            else: # pass through on miss
                colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1

    else:
        raise ValueError(f"recieved bad kwarg colors={colors}")

    if (disp_names is None) or (disp_names is False):
        disp_names_dict = {str(key): str(key) for key in data_dict}

    elif isinstance(disp_names, dict):
        disp_names_dict = {}
        for key in data_dict:
            if key in disp_names:
                disp_names_dict[key] = disp_names[key]
            else: # pass through on miss
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

    if (v_range is None) or (v_range is False):  # v_range will override q_range if both are provided
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
            f"spine_mode should be 'open', 'tight', or 'all', not {spine_mode}"
        )

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

                elif isinstance(particular_range, tuple):
                    ax.fill_between(
                        [
                            ax.get_xticks()[0] - 1 / 2,
                            ax.get_xticks()[-1] + 1 / 2,
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
                        + "not {shaded_range}"
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

    if isinstance(y_label, str):
        ax.set_ylabel(y_label)

    if isinstance(title, str):
        ax.set_title(title)

    if isinstance(save_path, (os.PathLike, str)):
        fig.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)

    if show:
        fig.show()


def categorical_scatter(
    data: pd.DataFrame | dict[str, NDArray | list | pd.DataFrame],
    *y_c: tuple[NDArray | list | pd.DataFrame, NDArray | list | pd.DataFrame],
    colors: dict[str, Color] | str = "nb50",
    size: float | list[float] | str = "auto",
    disp_names: dict[str | int, str] | None = None,
    mix_mode: str = "z_shuffle",  # 'z_shuffle', 'ordered', ('file_mix'?)
    ax_dims: tuple[float, float] = (2.5, 2.5),
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
    show_grid: bool | str = False,  # can be "major", "minor", or "both"
    include_cat_legend: bool = True,
    # include_size_legend: bool = False, # not yet implemented
    size_to_area_func: Callable[[NDArray], NDArray] | None = None,
    save_path: (
        str | os.PathLike[str] | None
    ) = None,  # where to save the plot, which will be a .png file
    show: bool = True,
    **kwargs,
) -> None:
    """
    This function makes a scatter plot where the points each point is colored by
    one of a dicrete set of categories (e.g. clusters, conditions).

    feature ideas:
     - allow the use of data fram column names as axis labels.
    """
    if len(y_c) == 2:  # recieved 3 args -> interpret as x, y, and c.
        x_raw, y_raw = (
            np.array(data),
            np.array(y_c[0]),
        )
        if isinstance(y_c[1], pd.DataFrame):
            c_raw = y_c[1][y_c[1].columns[0]].to_list()  # type: ignore
        else:
            c_raw = list(y_c[1])

    elif (len(y_c) == 1) | (len(y_c) > 2):
        raise TypeError(
            f"Number of arguments must be 1 or 3 not {len(y_c) + 1}."
        )

    else:  # (len(y_c) == 0)
        if isinstance(data, dict):
            if ("x" in data) and ("y" in data) and ("c" in data):
                x_raw, y_raw, c_raw = data["x"], data["y"], data["c"]
            else:
                raise ValueError(
                    "data of type dict must have keys 'x', 'y', and 'c'."
                )

        elif isinstance(data, pd.DataFrame):
            if (
                ("x" in data.columns)
                and ("y" in data.columns)
                and ("c" in data.columns)
            ):
                data_dropna = data.dropna(axis="rows")  # type: ignore
                x_raw = data_dropna["x"]
                y_raw = data_dropna["y"]
                c_raw = list(data_dropna["c"])
            else:
                raise ValueError(
                    "data of type DataFrame must have columns" +
                    " 'x', 'y', and 'c'."
                )
        else:
            raise TypeError(
                f"Try providing dict or DataFrame with columns" +
                " 'x', 'y', and 'c'."
            )


    # get s_raw, a list of the sizes of each point on the plot.
    if isinstance(size, float):
        s_raw = size * np.ones(len(c_raw))
    elif isinstance(size, list):
        if len(size) == len(c_raw):
            s_raw = np.array(size)
        else:
            raise ValueError("If s is a list, it must be the same length as c.")
    elif size == "auto":
        uniform_s = (
            200 * np.prod(ax_dims) / (5 + len(c_raw)) * np.ones(len(c_raw))
        )
        if isinstance(data, (dict, pd.DataFrame)) and (len(y_c) == 0):
            if "s" in data:
                s_raw = np.array(data["s"])
            else:
                s_raw = uniform_s
        else:
            s_raw = uniform_s
    else:
        raise TypeError(
            f"bad kwarg size = {size}; must be list, float, or 'auto'."
        )

    if size_to_area_func is None:
        size_func = lambda x: x
    elif callable(size_to_area_func):
        size_func = size_to_area_func
    else:
        raise TypeError(
            "size_to_area_func should be callable or None, "
            + "Not {size_to_area_func}."
        )

    if not len(x_raw) == len(y_raw) == len(c_raw) == len(s_raw):
        raise ValueError(
            f"incompatible lengths:\n"
            + f"    len(x)={len(x_raw)}\n"
            + f"    len(y)={len(y_raw)}\n"
            + f"    len(c)={len(c_raw)}\n"
            + f"    len(s)={len(s_raw)}"
        )

    cat_set = set(c_raw)

    if colors == "tab20":
        colors_dict = tab20_colors  # like Matplotlib
        n_colors = 20

    elif (colors == "nb50") or (colors is False) or (colors is None):
        colors_dict = nb50_colors
        n_colors = 50

    elif isinstance(colors, dict):
        colors_dict = {}
        default_colors = nb50_colors
        default_colors_index = 0
        for key in cat_set:
            if key in colors:
                colors_dict[key] = tuple(colors[key])
            else: # pass through on miss
                colors_dict[key] = default_colors[
                    default_colors_index % len(default_colors)
                ]
                default_colors_index += 1
        n_colors = len(colors_dict)

    else:
        raise TypeError(f"bad kwarg colors = {colors}")

    if not all([cat in colors_dict for cat in c_raw]):
        # more work will be requred to map c to colors
        if all(isinstance(key, int) for key in colors_dict.keys()):
            try:
                # if all elements of c can be converted to integers, this will
                # run without an error:
                _ = {int(cat) for cat in c_raw}  # type: ignore

                cat_to_color_key = {
                    cat: int(cat) % n_colors  # type: ignore
                    for cat in set(c_raw)
                }
            except ValueError:
                # In this case, categories will be arbitrarily mapped to colors.
                # This could lead to inconsistent color coding between plots.
                cat_to_color_key = {
                    cat: i % n_colors for i, cat in enumerate(cat_set)
                }
                if not all(
                    [(key in colors_dict) for key in cat_to_color_key.values()]
                ):
                    raise ValueError(
                        "Color mapping error. kwarg colors ="
                        + f" {colors} may have be a dict with "
                        + "non-sequential integer keys that do not exactly match"
                        + f"the values of 'c', {cat_set}."
                    )
            except Exception as e:
                raise ValueError("Unexpected error in color mapping.") from e

    else:
        cat_to_color_key = {cat: cat for cat in colors_dict}

    if disp_names:
        if isinstance(disp_names, dict):
            cat_to_disp_name = {}
            for key in cat_set:
                if key in disp_names:
                    cat_to_disp_name[key] = disp_names[key]
                else: # pass through on miss
                    cat_to_disp_name[key] = key 
        else:
            raise TypeError(
                "If provided, disp names should be of type dict, " +
                f"not {type(disp_names)}"
            )

        label_to_disp_name = {}
        if not y_label is None:
            if y_label in disp_names:
                label_to_disp_name[y_label] = disp_names[y_label]
            else:
                label_to_disp_name[y_label] = y_label
        if not x_label is None:
            if x_label in disp_names:
                label_to_disp_name[x_label] = disp_names[x_label]
            else:
                label_to_disp_name[x_label] = x_label
        if not title is None:
            if title in disp_names:
                label_to_disp_name[title] = disp_names[title]
            else:
                label_to_disp_name[title] = title
    else:
        cat_to_disp_name = {c: str(c) for c in cat_set}
        label_to_disp_name = {}
        if not y_label is None:
            label_to_disp_name[y_label] = y_label
        if not x_label is None:
            label_to_disp_name[x_label] = x_label
        if not title is None:
            label_to_disp_name[title] = title

    if mix_mode == "z_shuffle":
        point_order = np.random.permutation(len(c_raw))
    elif mix_mode == "ordered":
        point_order = np.arange(len(c_raw))
    else:
        raise ValueError(
            f"mix_mode {mix_mode} not recognized; try 'z_shuffle' or 'ordered'"
        )

    # prepare arguments for plt.scatter()
    x_array = np.array(x_raw)[point_order]
    y_array = np.array(y_raw)[point_order]
    color_list = [
        colors_dict[cat_to_color_key[c_raw[i]]]  # type: ignore
        for i in point_order
    ]
    s_array = np.array(s_raw)[point_order]

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
                True, 
                which=show_grid, # type: ignore
                zorder=-1, 
                linewidth=0.5
            )
        else:
            ax.grid(True, zorder=-1, linewidth=0.5)

    scatter_kwargs = {"marker", "alpha", "edgecolors"}

    if mix_mode in {"z_shuffle", "ordered"}:
        ax.scatter(
            x_array,
            y_array,
            c=color_list,
            s=size_func(s_array),
            linewidths=0,
            **{
                key: val  # type: ignore
                for key, val in kwargs.items()
                if key in scatter_kwargs
            },
        )
    else:  # If a mode was added to the previous mix_mode check but not here.
        raise ValueError(
            f"mix_mode {mix_mode} not recognized. Try 'z_shuffle' or 'ordered'"
        )

    if include_cat_legend:
        # create figure (not to ever be shown) to copy desired legend from
        legend_fig = plt.figure(figsize=(0.5, 0.5))
        legend_ax = legend_fig.add_axes((0, 0, 1, 1))
        for cat, key in cat_to_color_key.items():  # type: ignore
            if (cat in cat_to_disp_name) and (key in colors_dict):
                legend_ax.scatter(
                    0,
                    0,
                    c=colors_dict[key],  # type: ignore
                    label=cat_to_disp_name[cat],
                    linewidths=0,
                    **{
                        key: val  # type: ignore
                        for key, val in kwargs.items()
                        if key in scatter_kwargs ^ {"alpha"}
                    },
                )
        handles, labels = legend_ax.get_legend_handles_labels()
        plt.close(legend_fig)
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            markerscale=2,
            fancybox=True,
        )

    if x_lim:
        ax.set_xlim(x_lim)

    if y_lim:
        ax.set_ylim(y_lim)

    if log_scale_x:
        ax.set_xscale("log")

    if log_scale_y:
        ax.set_yscale("log")

    if x_label:
        ax.set_xlabel(label_to_disp_name[x_label])

    if y_label:
        ax.set_ylabel(label_to_disp_name[y_label])

    if title:
        ax.set_title(label_to_disp_name[title], color=title_color)

    if isinstance(save_path, (str, os.PathLike)):
        fig.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)

    if show:
        fig.show()
