"""
This file defines a StylePacket class, which stores shared information used in
multiple plots. The idea is that, when generating several plots for a document, 
you might want to consistently use a particular font, background color, etc., 
and it would be inconvenient to specify all of those things each time. The 
StylePacket object conveniently stores that kind of information and provides
methods that wrap various Matplotlib functions.
"""

import os
import importlib

# Load plotomata component modules
try:
    from . import color_palettes, _utils

    importlib.reload(color_palettes)
    importlib.reload(_utils)

    from ._utils import PassthroughDict
    from .color_palettes import Color

except ImportError as ie:
    # Alternative import style for non-standard import.
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils

        importlib.reload(color_palettes)
        importlib.reload(_utils)

        from _utils import PassthroughDict
        from color_palettes import Color

    except ImportError:
        raise ImportError(
            "plotomata failed to import component modules."
        ) from ie

    except Exception as e:
        raise ImportError(
            "Unexpected error while importing plotomata component modules."
        ) from e

except Exception as e:
    raise ImportError from e

from dataclasses import dataclass
from collections.abc import Hashable


@dataclass
class LabelGroup:
    """
    
    """
    keys: list[str] # definitive ordered list of the members of the label group.
    display_names: dict[str, str]
    colors: dict[str, Color]


class StylePacket:
    label_groups: set[LabelGroup]