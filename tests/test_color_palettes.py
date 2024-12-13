import warnings
import pytest
import numpy as np
from plotomata.color_palettes import Color


@pytest.fixture
def transparent_magenta():
    return Color(255, 0, 255, 0)


def test_Color_input_types():
    a = Color([1, 2, 3])
    b = Color((1, 2, 3, 255))
    c = Color("#010203ff")
    d = Color("0x010203")
    e = Color(1 / 255, 2 / 255, 3 / 255, 1.0)
    f = Color(np.array((1 / 255, 2 / 255, 3 / 255, 1.0)))
    g = Color(np.array([1, 2, 3, 255]))

    assert a == b == c == d == e == f == g


def test_Color_hsv_adjust(transparent_magenta):
    assert transparent_magenta.adjust_hue(
        0.5
    ) == transparent_magenta.adjust_hue(-0.5)
    assert transparent_magenta.adjust_hue(
        0.5
    ) != transparent_magenta.adjust_hue(-0.6)
    assert transparent_magenta.adjust_saturation(-0.5) == Color(
        (1.0, 0.5, 1.0, 0.0)
    )
    assert transparent_magenta / 2 == transparent_magenta * 0.5


def test_Color_properties(transparent_magenta):
    with pytest.raises(Exception):
        # Colors are immutable
        transparent_magenta.alpha = 1.0

    # Colors are tuples
    assert isinstance(transparent_magenta, tuple)
