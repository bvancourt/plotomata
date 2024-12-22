"""
This module defines standard sets of colors and other color-related code.
"""

import importlib
import os

# Load plotomata component modules
try:
    from . import _utils

    importlib.reload(_utils)

    from ._utils import is_tuple_as_string

except ImportError as ie:
    # Alternative import style for non-standard import (source_reticulate.py).
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])

        import _utils

        importlib.reload(_utils)

        from _utils import is_tuple_as_string

    except ImportError:
        raise ImportError(
            "plotomata failed to import component modules.\n"
        ) from ie

    except Exception as e:
        raise ImportError(
            "Unexpected error while importing plotomata component modules.\n"
        ) from e

except Exception as e:
    raise ImportError from e


import colorsys
from collections.abc import Iterable
import os
import re
import ast
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


PossibleColor = (  # types that might be interpretable as a color
    tuple[float, float, float, float]  # (r, g, b, a)
    | tuple[float, float, float]  # (r, g, b)
    | tuple[int, int, int, int]  # (r, g, b, a)
    | tuple[int, int, int]  # (r, g, b)
    | list[float | int]  # [r, g, b] or [r, g, b, a]
    | NDArray  # e.g. np.array((r, g, b))
    | str  # e.g. #ffffff
)


class Color(tuple):
    """
    Color is essentially a tuple[float, float, float, float] (r, g, b, a), but
    with additional functionality specific to colors. The elements can also be
    referenced by name, similar to a typing.NamedTuple, but overwriting __new__
    of that class caused an error.
    """

    def __new__(cls, *args: PossibleColor | int | float | np.ndarray):
        if len(args) == 1:
            # One arg was passed, which should contain r, g, and b values.
            possible_color = args[0]
        else:
            # Hopefully r, g, b (and optionally a) were passed as 3 or 4 args.
            possible_color = args

        if not isinstance(possible_color, Iterable):
            raise TypeError(
                "Only an iterable can be converted to Color, "
                + f"not {type(possible_color)}."
            )

        if len(possible_color) == 3:
            r_in, g_in, b_in = tuple(possible_color)
            # alpha should not be transparent if not provided
            if isinstance(r_in, (int, np.integer)):
                a_in = 255
            elif isinstance(r_in, (float, np.floating)):
                a_in = 1.0
            else:
                raise TypeError(
                    f"{args=} could not be converted to a Color. Must be 3 or 4"
                    + " numbers of the same type or hex string (e.g. #ffffff)."
                    + "\n"
                )
        elif len(possible_color) == 4:
            r_in, g_in, b_in, a_in = tuple(possible_color)

        elif isinstance(possible_color, str):
            # idea: support tuple as string? e.g. "(255, 0, 128, 255)"
            stripped_str = possible_color.removeprefix("#").removeprefix("0x")

            if bool(re.fullmatch(r"^[0-9A-Fa-f]+$", stripped_str)):
                if len(stripped_str) == 6:  # r, g, b in hex
                    r_in = int(stripped_str[0:2], 16)
                    g_in = int(stripped_str[2:4], 16)
                    b_in = int(stripped_str[4:6], 16)
                    a_in = 255
                elif len(stripped_str) == 8:  # r, g, b, a in hex
                    r_in = int(stripped_str[0:2], 16)
                    g_in = int(stripped_str[2:4], 16)
                    b_in = int(stripped_str[4:6], 16)
                    a_in = int(stripped_str[6:8], 16)
            elif is_tuple_as_string(possible_color):
                tuple_from_string = ast.literal_eval(possible_color)
                if len(tuple_from_string) in {3, 4}:
                    return Color(*tuple_from_string)
                else:
                    raise ValueError(
                        f"Possible color {possible_color} was interpreted as a "
                        + "string representation of a tuple, but it had the wr"
                        + f"ong length ({len(tuple_from_string)}, not 3 or 4)."
                    )
            else:
                raise TypeError(
                    f"str {possible_color} could not be interpreted as a color."
                    + "\n"
                )
        else:
            raise TypeError(
                f"{possible_color} of type {type(possible_color)} could not be "
                + "interpreted as a color.\n"
            )

        if (
            isinstance(r_in, (int, np.integer))
            and isinstance(g_in, (int, np.integer))
            and isinstance(b_in, (int, np.integer))
            and isinstance(a_in, (int, np.integer))
        ):
            red = np.clip(r_in, 0, 255) / 255
            green = np.clip(g_in, 0, 255) / 255
            blue = np.clip(b_in, 0, 255) / 255
            alpha = np.clip(a_in, 0, 255) / 255
        elif (
            isinstance(r_in, (float, np.floating))
            and isinstance(g_in, (float, np.floating))
            and isinstance(b_in, (float, np.floating))
            and isinstance(a_in, (float, np.floating))
        ):
            red = np.clip(r_in, 0, 1)
            green = np.clip(g_in, 0, 1)
            blue = np.clip(b_in, 0, 1)
            alpha = np.clip(a_in, 0, 1)
        else:
            raise TypeError(
                f"(r, g, b, a) = ({r_in}, {g_in}, {b_in}, {a_in}) extracted "
                f"from *{args=} could not be converted to a Color. These "
                "must be either all ints or all floats.\n"
            )

        self = super().__new__(cls, (red, green, blue, alpha))
        return self

    @property
    def red(self):
        return self[0]

    @property
    def green(self):
        return self[1]

    @property
    def blue(self):
        return self[2]

    @property
    def alpha(self):
        return self[3]

    def __eq__(self, other, eps: float = 0.001):
        if isinstance(other, Color):
            return (
                (np.abs(self.red - other.red) < eps)
                and (np.abs(self.green - other.green) < eps)
                and (np.abs(self.blue - other.blue) < eps)
                and (np.abs(self.alpha - other.alpha) < eps)
            )
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash((self.red, self.green, self.blue, self.alpha))

    @property
    def to_hsv(self):
        h, s, v = colorsys.rgb_to_hsv(self.red, self.green, self.blue)
        return h, s, v / 255

    def adjust_saturation(self, sat_diff: float):
        h, s, v = self.to_hsv
        r, g, b = colorsys.hsv_to_rgb(
            h, np.clip(s + sat_diff, 0.0, 1.0), v * 255
        )
        return Color(float(r), float(g), float(b), self.alpha)

    def adjust_hue(self, hue_diff: float):
        h, s, v = self.to_hsv
        r, g, b = colorsys.hsv_to_rgb((h + hue_diff) % 1.0, s, v * 255)
        return Color(float(r), float(g), float(b), self.alpha)

    def adjust_alpha(self, alpha: float):
        return Color(self.red, self.green, self.blue, self.alpha + alpha)

    def __mul__(self, number):
        if isinstance(number, (float, int)):
            return Color(
                self.red * number,
                self.green * number,
                self.blue * number,
                self.alpha,
            )
        else:
            return super().__mul__(number)

    def __truediv__(self, number):
        if isinstance(number, (float, int)):
            return Color(
                self.red / number,
                self.green / number,
                self.blue / number,
                self.alpha,
            )
        else:
            raise TypeError(f"Attempted to divide a Color by {number}.\n")

    def __repr__(self):
        return (
            f"Color(red={self.red}, green={self.green}, blue={self.blue}, "
            + f"alpha={self.alpha})"
        )


# Matplotlib default colors:
_mpl_tab20 = plt.get_cmap("tab20")
_tab20_order = np.array(
    [i * 2 for i in range(10)] + [i * 2 + 1 for i in range(10)]
)
tab20_colors: dict[int, Color] = [
    Color(_mpl_tab20(i_old)) for _, i_old in enumerate(_tab20_order)
]


# Custom colors used for nanobubble paper:
_npy_nb50 = np.clip(
    np.load(
        os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "color_data",
            "nb50.npy",
        )
    )
    / 255,
    0,
    1,
)

_nb50_order = np.arange(1, 51) % 50
nb50_colors: dict[int, Color] = [
    Color(_npy_nb50[i_old, 0], _npy_nb50[i_old, 1], _npy_nb50[i_old, 2], 1.0)
    for _, i_old in enumerate(_nb50_order)
]
