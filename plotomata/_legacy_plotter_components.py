"""
Additional file of code used by legacy_plotters.py
"""

import importlib

try:
    from . import color_palettes, _utils

    importlib.reload(color_palettes)
    from .color_palettes import Color
    from ._utils import PassthroughDict
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils

        importlib.reload(color_palettes)
        from color_palettes import Color
        from _utils import PassthroughDict
    except ImportError as ie2:
        raise ie2 from ie
except Exception as e:
    raise ImportError from e

from typing import Callable, Hashable
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.artist import Artist
import pandas as pd
from pandas.core.series import Series
from numpy.typing import NDArray


Arrayable = NDArray | list | pd.DataFrame | Series


def standardize_arrayable(
    arrayable: Arrayable,
    dtype: str | type = "auto",
) -> NDArray | list:
    """
    There are a number of argument types that might, for various reasons, have
    been passed to basically represent an array or list of data values. This
    function contains code to check input types and convert to.

    dtype kwarg can be used to specify the dtype for numpy array output. "auto"
    will give a NumPy array of whatever type automatically comes form the input,
    or "list" will give a list instead of an array.
    """
    if isinstance(arrayable, pd.DataFrame):  # will take only the first column
        if dtype == "list":
            return arrayable.dropna(axis="rows")[  # type: ignore
                arrayable.columns[0]
            ].to_list()
        elif dtype == "auto":
            return np.array(
                arrayable.dropna(axis="rows")[  # type: ignore
                    arrayable.columns[0]
                ]
            )
        elif isinstance(dtype, type):
            return np.array(
                arrayable.dropna(axis="rows")[  # type: ignore
                    arrayable.columns[0]
                ],
                dtype=dtype,
            )
        else:
            raise TypeError(
                "kwarg 'dtype' should be a type or 'list' or 'auto', not"
                + f" {dtype}.\n"
            )
    elif isinstance(arrayable, (np.ndarray, Series, list)):
        if dtype == "list":
            return list(arrayable)
        elif dtype == "auto":
            return np.array(arrayable)
        elif isinstance(dtype, type):
            return np.array(arrayable, dtype=dtype)
        else:
            raise TypeError(
                "kwarg 'dtype' should be a type or 'list' or 'auto', not"
                + f" {dtype}.\n"
            )
    else:
        raise TypeError(
            "arg 'arrayable' must have type pd.DataFrame, np.ndarray, Series, "
            + f"or list, not {type(arrayable)}.\n"
        )


def label_axes_etc(
    ax: Axes,
    ax_dims: tuple[float, float] | None = None,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    disp_names: dict[str, str] | None = None,
    title_color: Color | str = (0, 0, 0, 1),
    x_label_color: Color | str = (0, 0, 0, 1),
    y_label_color: Color | str = (0, 0, 0, 1),
    hide_spines: bool | list[str] = False,
    aspect_ratio: float | None = None,  # "equal", "auto", or float(dy/dx)
    axis_off: bool = False,
    **miss_kwargs,
) -> Axes:
    """
    Setting the axis labels, limits, title, etc. is necessary for most plots, so
    standard code for it is prvided here.
    """
    if disp_names is None:
        disp_names = PassthroughDict({})

    if x_lim:
        ax.set_xlim(x_lim)

    if y_lim:
        ax.set_ylim(y_lim)

    if log_scale_x:
        ax.set_xscale("log")

    if log_scale_y:
        ax.set_yscale("log")

    if x_label:
        ax.set_xlabel(disp_names[x_label], color=x_label_color)

    if y_label:
        ax.set_ylabel(disp_names[y_label], color=y_label_color)

    if title:
        ax.set_title(disp_names[title], color=title_color)

    if hide_spines:
        if isinstance(hide_spines, list):
            for spine in hide_spines:
                ax.spines[spine].set_visible(False)
        else:
            ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    if axis_off:
        ax.set_axis_off()

    if aspect_ratio:
        if ax_dims is None:
            raise NotImplementedError(
                "Please do not provide aspect_ratio without providing ax_dims."
            )
        if x_lim and y_lim:
            raise ValueError(
                "Attempted to specify x_lim, y_lim, and aspect for existing "
                + "axes, which is not allowed, since label_axes_etc() is not "
                + "allowed to change the axes 'position'."
            )
        elif isinstance(aspect_ratio, (float, int)):
            target_ar = float(aspect_ratio)
        elif aspect_ratio == "equal":
            target_ar = 1
        elif aspect_ratio == "auto":
            pass
        else:
            raise ValueError(
                f"'aspect_ratio' should be a float or 'equal', not "
                + "{aspect_ratio}."
            )

        ax_width, ax_height = ax_dims
        current_x_min, current_x_max = ax.get_xlim()
        current_y_min, current_y_max = ax.get_ylim()

        current_y_per_inch = (current_y_max - current_y_min) / ax_height
        current_x_per_inch = (current_x_max - current_x_min) / ax_width

        current_ar = current_y_per_inch / current_x_per_inch

        if y_lim or (current_ar > target_ar):
            # increase diff(xlim) to reduce aspect ratio
            x_center = (current_x_min + current_x_max) / 2
            target_x_per_inch = current_y_per_inch / target_ar
            ax.set_xlim(
                x_center - target_x_per_inch * ax_width / 2,
                x_center + target_x_per_inch * ax_width / 2,
            )

        else:
            # increase diff(ylim) to increase aspect ratio
            y_center = (current_y_min + current_y_max) / 2
            target_y_per_inch = current_x_per_inch * target_ar
            ax.set_ylim(
                y_center - target_y_per_inch * ax_width / 2,
                y_center + target_y_per_inch * ax_width / 2,
            )

    return ax


def surrogate_color_legend_info(
    colors: dict[str | int, Color],
    disp_names: dict[str, str] | None = None,
    **kwargs,
) -> tuple[list[Artist], list[str]]:
    """
    This function makes a dummy figure just to get handles and labels for
    another figure's legend. This can be helpful if that figure was constructed
    generated in such a way that is isn't the labels are wrong and you need to
    maunally fix it.
    """
    if disp_names is None:
        disp_names = PassthroughDict({})

    # create figure (not to ever be shown) to copy desired legend from
    legend_fig = plt.figure(figsize=(0.5, 0.5))
    legend_ax = legend_fig.add_axes((0, 0, 1, 1))

    for key in colors:
        legend_ax.scatter(
            0,
            0,
            color=colors[key],  # type: ignore
            label=disp_names[key],  # type: ignore
            linewidths=0,
            **kwargs,
        )

    handles, labels = legend_ax.get_legend_handles_labels()
    plt.close(legend_fig)
    return handles, labels


def surrogate_size_legend_info_preselected(
    sizes: Arrayable, labels: Arrayable, **scatter_kwargs
) -> tuple[list[Artist], list[str]]:
    """
    This function makes a dummy figure just to get handles and labels for
    another figure's legend. Specifically, this is used for marker size legends.
    """
    s_array = standardize_arrayable(sizes)
    labels_list = standardize_arrayable(labels, dtype="list")

    if not len(s_array) == len(labels_list):
        raise ValueError(
            "Sizes and lables must have the same number of elements."
            + f"detected {len(sizes)} sizes and {len(labels)} labels.\n"
        )

    legend_fig = plt.figure(figsize=(0.5, 0.5))
    legend_ax = legend_fig.add_axes((0, 0, 1, 1))
    for size, label in zip(s_array, labels_list):
        legend_ax.scatter(
            0,
            0,
            s=size,
            label=label,
            linewidths=0,
            color="black",
            **scatter_kwargs,
        )

    handles, labels = legend_ax.get_legend_handles_labels()
    plt.close(legend_fig)
    return handles, labels


def surrogate_size_legend_info_automatic(
    size_data: Arrayable,
    size_to_area_func: Callable | None = None,
    n_labels: int = 4,
    max_digits: int = 4,
    **scatter_kwargs,
) -> tuple[list[Artist], list[str]]:
    """
    This is basically a wrapper for surrogate_size_legend_info_preselected()
    that automatically figures out which sizes to label instead of requiring
    them to be listed in the args.
    """
    if (max_digits < 1) or not isinstance(max_digits, int):
        raise ValueError(
            f"Invalid kwarg max_digits={max_digits}. Should be an int >=1.\n"
        )
    if (n_labels < 2) or not isinstance(n_labels, int):
        raise ValueError(
            f"Invalid kwarg n_labels={n_labels}. Should be an integer >=2.\n"
        )
    if size_to_area_func is None:
        area_func = lambda x: x
    elif isinstance(size_to_area_func, Callable):
        area_func = size_to_area_func
    raw_size_array = standardize_arrayable(size_data)

    quant_sizes = np.quantile(
        raw_size_array, np.arange(n_labels) / (n_labels - 1)
    )

    lin_sizes = np.linspace(
        np.min(raw_size_array), np.max(raw_size_array), n_labels
    )

    log_sizes = np.exp(
        np.linspace(
            np.log(np.min(raw_size_array)),
            np.log(np.max(raw_size_array)),
            n_labels,
        )
    )

    comp_sizes = (quant_sizes + lin_sizes + log_sizes) / 3

    best_quality = 0
    best_sizes = np.array(
        list(
            set(
                [
                    np.round(
                        s, int(max_digits - 1 - np.round(np.log10(np.abs(s))))
                    )
                    for s in comp_sizes[~np.isinf(comp_sizes)]
                ]
            )
        )
    )
    for candidate_sizes in [quant_sizes, lin_sizes, log_sizes, comp_sizes]:
        for n in range(max_digits):
            rounded_sizes = np.array(
                list(
                    set(
                        [
                            np.round(s, int(n - np.round(np.log10(np.abs(s)))))
                            for s in candidate_sizes
                        ]
                    )
                ),
                dtype=raw_size_array.dtype,
            )  # type: ignore
            if len(rounded_sizes) == n_labels:
                break
        else:
            raise RuntimeWarning(
                "surrogate_size_legend_info_automatic() failed to find a good "
                + f"set of {n_labels} sizes to label. "
            )
        rounded_sizes = np.sort(
            rounded_sizes[~np.isnan(rounded_sizes) & ~np.isinf(rounded_sizes)]
        )
        if len(rounded_sizes) > 2:
            candidate_quality = 1 / np.max(
                np.diff(np.sqrt(area_func(rounded_sizes)))
            )
        else:
            candidate_quality = 0
        if candidate_quality > best_quality:
            best_quality = candidate_quality
            best_sizes = rounded_sizes

    actual_areas = area_func(best_sizes)
    valid = (
        (actual_areas > 0) & ~np.isnan(actual_areas) & ~np.isinf(actual_areas)
    )
    actual_labels = [str(s) for s in best_sizes[valid]]
    i_sort = np.argsort(actual_areas[valid])

    return surrogate_size_legend_info_preselected(
        actual_areas[valid][i_sort],
        [actual_labels[i] for i in i_sort],
        **scatter_kwargs,
    )


def labels_from_data(
    *args,
    x_label: str | None = None,
    y_label: str | None = None,
    color_data_label: str | None = None,
    size_legend_title: str | None = None,
):
    labels_dict = {
        "x_label": x_label,
        "y_label": y_label,
        "color_data_label": color_data_label,
        "size_legend_title": size_legend_title,
    }

    if isinstance(args[0], pd.DataFrame):
        if (len(args[0].columns) >= 3) and not all(
            ("x" in args[0], "y" in args[0], "c" in args[0])
        ):
            for i in range(3):
                labels_dict[list(labels_dict)[i]] = args[0].columns[i]

    for i, label_key in enumerate(labels_dict):
        if len(args) > i:
            if isinstance(args[i], Series):
                labels_dict[label_key] = args[i].name

    return labels_dict
