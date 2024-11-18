"""
This module will store miscelaneous small bits of useful code.
"""

import importlib

try:
    from . import color_sets

    importlib.reload(color_sets)
    from .color_sets import Color
except ImportError as ie:
    # normal import style above may not work with reticulate_source.py
    try:
        import os
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_sets

        importlib.reload(color_sets)
        from color_sets import Color
    except ImportError as ie2:
        raise ie2 from ie
except Exception as e:
    raise ImportError from e

from typing import TypeAlias, Callable, Hashable
from functools import cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.artist import Artist
import pandas as pd
from pandas.core.series import Series
from numpy.typing import NDArray

Arrayable: TypeAlias = NDArray | list | pd.DataFrame | Series


class PassthroughDict(dict):
    """
    Slightly modified dictionary that returns the key rather than raising an
    an error when you try to get an item with a key it doesn't have. This is
    useful for the "disp_names" dictionaries used in various plotter functions.
    """

    def __missing__(self, key):
        return key


def invert_dictionary(
    dictionary: dict[Hashable, Hashable],
    passthrough: bool = False,
):
    if passthrough:
        return PassthroughDict(
            {value: key for key, value in dictionary.items()}
        )
    else:
        return {value: key for key, value in dictionary.items()}


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