"""
This module defines standard sets of colors and types used to store colors.
"""

import os
from typing import TypeAlias
import numpy as np
import matplotlib.pyplot as plt


# name some type aliases
Color: TypeAlias = tuple[float, float, float, float]  # (r, g, b, a)
ListColor: TypeAlias = list[float]  # [r, g, b, a]; (for getting colors form R)

# Matplotlib default colors:
_mpl_tab20 = plt.get_cmap("tab20")
_tab20_order = np.array(
    [i * 2 for i in range(10)] + [i * 2 + 1 for i in range(10)]
)
tab20_colors: dict[int, Color] = {
    i_new: _mpl_tab20(i_old) for i_new, i_old in enumerate(_tab20_order)
}

# Custom colors used for nanobubble paper:
_npy_nb50 = np.clip(
    np.load(
        os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "color_sets",
            "cmap50_grayhead.npy",
        )
    )
    / 255,
    0,
    1,
)
_nb50_order = np.arange(1, 51) % 50
nb50_colors: dict[int, Color] = {
    i_new: (_npy_nb50[i_old, 0], _npy_nb50[i_old, 1], _npy_nb50[i_old, 2], 1.0)
    for i_new, i_old in enumerate(_nb50_order)
}
