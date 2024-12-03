"""
This module holds miscellaneous utilities that are used by other modules but do
not clearly fall within their scope.
"""

import os
import importlib

# Load plotomata component modules
try:
    from . import color_palettes, _utils

    importlib.reload(color_palettes)
    importlib.reload(_utils)

    from ._utils import (
        PassthroughDict,
        wrap_transformation_func,
        all_are_instances,
    )
    from .color_palettes import Color

except ImportError as ie:
    # Alternative import style for non-standard import.
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils

        importlib.reload(color_palettes)
        importlib.reload(_utils)

        from _utils import (
            PassthroughDict,
            wrap_transformation_func,
            all_are_instances,
        )
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

from typing import Callable
from dataclasses import dataclass
from enum import Enum, auto
import warnings
from collections.abc import Hashable
import pandas as pd
from pandas.core.series import Series
from pandas.core.indexes.base import Index
import numpy as np
from numpy.typing import NDArray



class CommandNames(Enum):
    """
    These are the possible ParsingCommands. See also _process_string().
    """
    GLOBAL = auto()
    IGNORE = auto()
    READ_AS_COLOR = auto()


@dataclass(slots = True)
class ParserWord:
    """
    These are the "words" of the language that the arg_parser() function speaks 
    to the plotter functions. This is a base class not intended for direct use.
    """
    # Indices of the source of this ParserWord in args. It is a list[int]
    #   because there can be sub-indices into some individual args.
    arg_indices: list[int]

    def assert_validity(self):
        if (
            isinstance(self.arg_indices, list)
            and (len(self.arg_indices) > 0)
            and all(isinstance(elem, int) for elem in self.arg_indices)
        ):
            return True
        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid ParserWord."
            )

    def __post_init__(self):
        _ = self.assert_validity()


class MalformedParserWord(Exception):
    """
    Exception to be raised if a ParserWord is found to be invalid.
    """
    pass


@dataclass(slots = True)
class Numbers(ParserWord):
    """
    Numerical data, e.g. for coordinates, colors, or sizes of points on a
    scatter plot.
    """
    numbers: (
        dict[Hashable, float | int]
        | NDArray[np.floating | np.integer]
        | set[float | int]
    )
    name: Hashable | None = None

    def __len__(self):
        return len(self.numbers)
    
    def assert_validity(self):
        if (
            ParserWord.assert_validity(self)
            and isinstance(self.name, Hashable)
            and isinstance(self.numbers, (dict, np.ndarray, set))
            and all_are_instances(
                self.numbers, 
                (int, float, np.integer, np.floating)
            )
        ):
            return True
        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid Numbers(ParserWord)."
            )
    
    def to_labels_if_possible(
            self,
            check_types: bool = True, 
            self_if_impossible: bool = False,
        ):
        if check_types:
            # probably only int or bool should be converted to labels
            match self.numbers:
                case dict():
                    if not all((
                        isinstance(value, (int, np.integer, bool, np.bool_))
                        for value in self.numbers.values()
                    )):
                        return self
                case np.ndarray():
                    if not np.issubdtype(self.number.dtype, np.integer):
                        return self
                case set():
                    if not all((
                        isinstance(value, (int, np.integer, bool, np.bool_))
                        for value in self.numbers
                    )):
                        return self
                case _:
                    raise MalformedParserWord(
                        "Numbers.to_labels_if_possible() type check found that "
                        + "numbers attribute that was not dict, set, or NDArray"
                    )
    
    def to_color_if_possible(
            self, 
            clip : bool = False, 
            return_self_if_not_color : bool = True,
        ):
        if (len(self) in [3, 4]) and isinstance(self.numbers, np.ndarray):
            if np.issubdtype(self.numbers.dtype, np.floating):
                if clip:
                    return Colors(
                        arg_indices = self.arg_indices,
                        name = self.name,
                        colors = [Color(np.clip(self.numbers, 0, 1))]
                    )
                elif np.all((self.numbers >= 0) & (self.numbers <= 1)):
                    return Colors(
                        arg_indices = self.arg_indices,
                        name = self.name,
                        colors = [Color(self.numbers)]
                    )
            elif np.issubdtype(self.numbers.dtype, np.integer):
                if clip:
                    return Colors(
                        arg_indices = self.arg_indices,
                        name = self.name,
                        colors = [Color(np.clip(self.numbers, 0, 255))]
                    )
                elif np.all(self.numbers // 256 == 0):
                    return Colors(
                        arg_indices = self.arg_indices,
                        name = self.name,
                        colors = [Color(self.numbers)]
                    )

        if return_self_if_not_color:
            return self


@dataclass(slots = True)
class Labels(ParserWord):
    """
    Labels could be used to label plots, axes, categories, etc., and could also
    be keys used internally to link up data with colors and other labels. Most
    often these would be strings, but other Hashables are also supported.
    """
    labels: dict[Hashable, Hashable] | list[Hashable] | set[Hashable]
    name: Hashable = None

    def __len__(self):
        return len(self.labels)
    
    def __post_init__(self):
        if isinstance(self.labels, dict):
            self.labels = PassthroughDict(self.labels)
        self.assert_validity()

    def assert_validity(self):
        if (
            ParserWord.assert_validity(self)
            and isinstance(self.name, Hashable)
            and isinstance(self.labels, (PassthroughDict, list, set))
            and all_are_instances(self.labels, Hashable)
        ):
            return True

        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid Labels(ParserWord)."
            )


@dataclass(slots = True)
class Colors(ParserWord):
    """
    Literally colors to be used in plots.
    """
    colors: dict[Hashable, Color] | list[Color] | set[Color]
    name: Hashable = None

    def __len__(self):
        return len(self.colors)


@dataclass(slots = True)
class ParsingCommand(ParserWord):
    """
    These carry instructions about how other ParserWords should be interpreted.
    """
    command: CommandNames


@dataclass(slots = True)
class Transformation(ParserWord):
    """
    These can be used to modify Numbers items for visual representation. For
    example, a transformation may be desirable to make the radius of dots on a
    scatter plot proportional to some data, rather than the area (default), or
    arbitrary functions like [area = -log(x + 1)], etc. Axes, size legend, etc.
    Will be labeled using the un-transformed data values.
    """
    func: Callable


def _process_data_frame(
        arg_index : int, 
        data_frame : pd.DataFrame
    ) -> list[ParserWord]:

    parser_words = []
    for col_index, col_key in enumerate(data_frame.columns):
        if pd.api.types.is_numeric_dtype(data_frame[col_key].dtype):
            parser_words.append(Numbers(
                arg_indices = [arg_index, col_index],
                numbers = np.array(data_frame[col_key]),
                name = col_key,
            ))
        else:
            parser_words.append(Labels(
                arg_indices = [arg_index, col_index],
                labels = list(data_frame[col_key]),
                name = col_key,
            ))

    if type(data_frame.index) == Index:
        # The index of a DataFrame created from a dict without subsequently
        #   setting this will have a different type, and is ignored.
        parser_words.append(Labels(
            arg_indices = [arg_index, len(data_frame.columns)],
            labels = list(data_frame.index),
        ))
    
    return parser_words


def _process_series(arg_index : int, series: Series) -> list[ParserWord]:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return [Numbers(
            arg_indices = [arg_index],
            numbers = np.array(series),
            name = series.name,
        )]
    else:
        return [Labels(
            arg_indices = [arg_index],
            labels = list(series),
            name = series.name,
        )]

def _process_array(arg_index : int, array: NDArray) -> list[ParserWord]:
    return [Numbers(
        arg_indices = [arg_index],
        numbers = array,
    )]

def _process_dict(arg_index : int, dictionary: dict) -> list[ParserWord]:
    if all_are_instances(dictionary.values(), (int, float)):
        return [Numbers(
            arg_indices = [arg_index],
            numbers = dictionary,
        )]
    elif all_are_instances(dictionary.values(), Hashable):
        return [Labels(
            arg_indices = [arg_index],
            labels = dictionary,
        )]
    elif all_are_instances(dictionary.values(), Color):
        return [Colors(
            arg_indices = [arg_index],
            labels = dictionary,
        )]
    else: # dict could be used to specify names for ParserWords
        parser_words = []
        for name, value in dictionary.items():
            new_words = _single_arg_to_parser_words(value, arg_index)
            for word in new_words:
                if hasattr(word, "name"):
                    word.name = name
                word.arg_indices = [arg_index] + word.arg_indices
                parser_words.append(word)
        return parser_words

def _process_list(arg_index : int, elements) -> list[ParserWord]:
    if all_are_instances(elements, (int, float)):
        return [Numbers(
            arg_indices = [arg_index],
            numbers = np.array(elements),
        )]

    elif all_are_instances(elements, Hashable):
        if all((elem in CommandNames.__members__ for elem in elements)):
            return [
                ParsingCommand(
                    arg_indices = [arg_index], 
                    command = CommandNames[elem]
                )
                for elem in elements
            ]

        else:
            return [Labels(
                arg_indices = [arg_index],
                labels = elements,
            )]

def _process_tuple(arg_index : int, tuple_arg) -> list[ParserWord]:
    if isinstance(tuple_arg, Color):
        return [Colors(
            arg_indices = [arg_index],
            colors = {tuple_arg},
        )]
    else:
        # If the tuple is not a color, it gets unpacked into individual
        # other types, except that they all have the same first arg_indices,
        # causing any parsing commands or transformations referring to that
        # index to modify all of them.

        parser_words = _args_to_parser_words(*tuple_arg)
        for elem in parser_words:
            # prepend outer index
            elem.arg_indices.insert(0, arg_index)

        return parser_words

def _process_string(arg_index: int, string: str) -> list[ParserWord]:
    if string in CommandNames.__members__:
        return([ParsingCommand(
            arg_indices = [arg_index], 
            command = CommandNames[string],
        )])
    else: # Interpreted as a single Label
        return([Labels(
            arg_indices = [arg_index], 
            labels = {string},
            name = string,
        )])

def _process_callable(arg_index: int, callable: Callable) -> list[ParserWord]:
    try:
        return [Transformation(
            arg_indices = [arg_index],
            func = wrap_transformation_func(callable),
        )]
    except Exception as e:
        warnings.warn(
            "Failed to create Transformation(ParserWord) from Callable at "
            + f"{arg_index=}. Exception: {str(e)}."
        )
        return []
    
def _single_arg_to_parser_words(arg, arg_index: int) -> list[ParserWord]:
    """
    This function does most of the work for _args_to_parser_words() (see that
    function's docstring for more explanation) and is also used directly by
    _process_dict(). It takes something passed to plotomata by a user and routes
    it to one of several function (depending on type) for conversion to
    list[ParserWord] format.
    """
    match arg:
        case pd.DataFrame():
            return _process_data_frame(arg_index, arg)
        case Series():
            return _process_series(arg_index, arg)
        case np.ndarray():
            return _process_array(arg_index, arg)
        case list():
            return _process_list(arg_index, arg)
        case dict():
            return _process_dict(arg_index, arg)
        case tuple():
            return _process_tuple(arg_index, arg)
        case str():
            return _process_string(arg_index, arg)
        case ParserWord():
            arg.arg_index = arg_index
            return [arg]
        case _ if callable(arg):
            return _process_callable(arg_index, arg)
        case _:
            warnings.warn(
                f"Unable to standardize argument {arg} of type {type(arg)}."
                + " Consider converting to DataFrame, Series, NDArray,"
                + " dict, string, Color, or list."
            )
            return []

def _args_to_parser_words(*args) -> list[ParserWord]:
    """
    This function takes the args passed to a plotting function, which may come
    in a variety of types, determines what they could be used for (e.g. a
    series of numbers, a colormap, display names, etc.), and produces a new list
    of the same information in the form of ParserWords. Each arg could produce
    several control items or none.
    """
    parser_words: list[ParserWord] = []
    for arg_index, arg in enumerate(args):
        parser_words += _single_arg_to_parser_words(arg, arg_index)

    return parser_words

def arg_parser(parsing_directive, *args):
    """
    This function converts arguments from the user into a ParserWord list, then
    analyzes the "grammar" of that list to determine which items modify others,
    and what purposes they should fill in making a plot.

    parsing_directive provides context about what to search for (e.g. )
    """
    parser_words = _args_to_parser_words(*args)
