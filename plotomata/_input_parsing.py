"""
This module holds miscellaneous utilities that are used by other modules but do
not clearly fall within their scope.
"""

import os
import importlib

# Load plotomata component modules
try:
    from . import color_palettes, _utils, style_packets

    importlib.reload(color_palettes)
    importlib.reload(_utils)
    importlib.reload(style_packets)

    from ._utils import (
        PassthroughDict,
        DefaultList,
        wrap_transformation_func,
        all_are_instances,
        possible_parser_scopes,
    )
    from .color_palettes import Color
    from .style_packets import SettingsPacket

except ImportError as ie:
    # Alternative import style for non-standard import.
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils, style_packets

        importlib.reload(color_palettes)
        importlib.reload(_utils)
        importlib.reload(style_packets)

        from _utils import (
            PassthroughDict,
            DefaultList,
            wrap_transformation_func,
            all_are_instances,
            possible_parser_scopes,
        )
        from color_palettes import Color
        from style_packets import SettingsPacket

    except ImportError:
        raise ImportError(
            "plotomata failed to import component modules."
        ) from ie

    except Exception as e:
        raise ImportError(
            "Unexpected error while importing plotomata component modules.\n"
        ) from e

except Exception as e:
    raise ImportError from e

import copy
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
    These are the possible Commands. See also _process_string().
    """

    GLOBAL = auto()
    IGNORE = auto()
    READ_AS_COLOR = auto()


@dataclass(slots=True)
class ParserWord:
    """
    These are the "words" of the language that the arg_parser() function speaks
    to the plotter functions. This is a base class not intended for direct use.
    """

    # Indices of the source of this ParserWord in args. It is a list[int]
    #   because there can be sub-indices into some individual args.
    arg_indices: DefaultList[int]

    def assert_validity(self):
        if (
            isinstance(self.arg_indices, DefaultList)
            and (len(self.arg_indices) > 0)
            and all(isinstance(elem, int) for elem in self.arg_indices)
        ):
            return True
        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid ParserWord.\n"
                + f"isinstance(self.arg_indices, DefaultList) = {isinstance(self.arg_indices, DefaultList)}\n"
                + f"{self.arg_indices=}\n"
                + f"arg_indices_types: {[type(elem) for elem in self.arg_indices]}"
            )

    def __post_init__(self):
        # make sure teh arg indices are integers.
        self.arg_indices = DefaultList(
            [int(index) for index in self.arg_indices]
        )

    def __len__(self):
        # This should be overwritten in child classes that store iterable data.
        return 1

    def could_refer_to(self, other):
        # This should be overwritten for child classes that can refer to others.
        return False


class MalformedParserWord(Exception):
    """
    Exception to be raised if a ParserWord is found to be invalid.
    """

    pass


class InvalidParserState(Exception):
    """
    Exception to be raised if a ParserState is found to be invalid.
    """

    pass


@dataclass(slots=True)
class Numeric(ParserWord):
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
    assert_valid: bool = True

    def __len__(self):
        return len(self.numbers)

    def could_refer_to(self, other):
        if isinstance(other, Numeric) and (len(self) == len(other)):
            return True
        else:
            return False

    def assert_validity(self):
        if (
            ParserWord.assert_validity(self)
            and isinstance(self.name, Hashable)
            and isinstance(self.numbers, (dict, np.ndarray, set))
            and all_are_instances(
                self.numbers, (int, float, np.integer, np.floating)
            )
            and isinstance(self.assert_valid, bool)
        ):
            return True
        elif self.assert_valid is False:
            warnings.warn(
                "Attempted to assert validity of an invalid Numeric(ParserWord)"
                + ". This did not raise an error because it also had attribute"
                + " assert_valid=False.\n"
            )
        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid Numeric(ParserWord)."
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
                    if not all(
                        (
                            isinstance(value, (int, np.integer, bool, np.bool_))
                            for value in self.numbers.values()
                        )
                    ):
                        return self
                case np.ndarray():
                    if not np.issubdtype(self.number.dtype, np.integer):
                        return self
                case set():
                    if not all(
                        (
                            isinstance(value, (int, np.integer, bool, np.bool_))
                            for value in self.numbers
                        )
                    ):
                        return self
                case _:
                    raise MalformedParserWord(
                        "Numeric.to_labels_if_possible() type check found that "
                        + "numbers attribute that was not dict, set, or NDArray"
                        + ".\n"
                    )

    def to_color_if_possible(
        self,
        clip: bool = False,
        return_self_if_not_color: bool = True,
    ):
        if (len(self) in [3, 4]) and isinstance(self.numbers, np.ndarray):
            if np.issubdtype(self.numbers.dtype, np.floating):
                if clip:
                    return Colors(
                        arg_indices=self.arg_indices,
                        name=self.name,
                        colors=[Color(np.clip(self.numbers, 0, 1))],
                    )
                elif np.all((self.numbers >= 0) & (self.numbers <= 1)):
                    return Colors(
                        arg_indices=self.arg_indices,
                        name=self.name,
                        colors=[Color(self.numbers)],
                    )
            elif np.issubdtype(self.numbers.dtype, np.integer):
                if clip:
                    return Colors(
                        arg_indices=self.arg_indices,
                        name=self.name,
                        colors=[Color(np.clip(self.numbers, 0, 255))],
                    )
                elif np.all(self.numbers // 256 == 0):
                    return Colors(
                        arg_indices=self.arg_indices,
                        name=self.name,
                        colors=[Color(self.numbers)],
                    )

        if return_self_if_not_color:
            return self


@dataclass(slots=True)
class Labels(ParserWord):
    """
    Labels could be used to label plots, axes, categories, etc., and could also
    be keys used internally to link up data with colors and other labels. Most
    often these would be strings, but other Hashables are also supported.
    """

    labels: dict[Hashable, Hashable] | list[Hashable] | set[Hashable]
    name: Hashable = None
    assert_valid: bool = True

    def __len__(self):
        return len(self.labels)

    def could_refer_to(self, other):
        if (len(self) == 1) and hasattr(other, "name") and (other.name is None):
            # self could be interpreted as a name for other.
            return True

        elif (
            isinstance(other, Numeric)
            and (  # self could be names for the elements of other.
                (len(self) == len(other))
            )
            and (
                (  # both are dictionaries with matching keys
                    isinstance(self.labels, dict)
                    and isinstance(other.numbers, dict)
                    and all(key in other.numbers for key in self.labels)
                )
                or (  # list can label a numeric array or dict, but not set.
                    isinstance(self.labels, list)
                    and not isinstance(other.numbers, set)
                )
            )
        ):
            return True

        elif isinstance(
            other,
            Labels,  # self could be a key correction or display names for other
        ) and (
            (
                isinstance(other.labels, list)
                and any(elem in self.labels for elem in other.labels)
            )
            or (
                isinstance(other.labels, dict)
                and any(elem in self.labels for elem in other.labels.values())
            )
        ):
            return True

        elif isinstance(
            self.labels,
            (set, list),
            # self could be a list of other words or elements to apply a
            #   command, transformation, or color to.
        ) and (
            (other.name in self.labels)
            or (
                isinstance(other, Numeric)
                and isinstance(other.numbers, dict)
                and any(self.labels in other)
            )
            or (
                isinstance(other, Labels)
                and isinstance(other.labels, dict)
                and any(self.labels in other)
            )
            or (
                isinstance(other, Colors)
                and isinstance(other.colors, dict)
                and any(self.labels in other)
            )
        ):
            return True
        else:
            return False

    def __post_init__(self):

        if isinstance(self.labels, dict):
            self.labels = PassthroughDict(self.labels)

        ParserWord.__post_init__(self)

        if self.assert_valid:
            self.assert_validity()

    def assert_validity(self):
        if (
            ParserWord.assert_validity(self)
            and isinstance(self.name, Hashable)
            and isinstance(self.labels, (PassthroughDict, list, set))
            and all_are_instances(self.labels, Hashable)
            and isinstance(self.assert_valid, bool)
        ):
            return True

        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid Labels(ParserWord).\n"
            )

    def to_color_if_possible(self):
        if isinstance(self.labels, dict):
            try:
                return Colors(
                    arg_indices=self.arg_indices,
                    colors={
                        key: Color(value) for key, value in self.labels.items()
                    },
                    name=self.name,
                )
            except:
                return self
        elif isinstance(self.labels, list):
            try:
                return Colors(
                    arg_indices=self.arg_indices,
                    colors=[Color(elem) for elem in self.labels],
                    name=self.name,
                )
            except:
                return self
        elif isinstance(self.labels, set):
            try:
                return Colors(
                    arg_indices=self.arg_indices,
                    colors={Color(elem) for elem in self.labels},
                    name=self.name,
                )
            except:
                return self
        else:
            raise MalformedParserWord(
                "Labels.labels should be either a list, set, or a dict, not "
                + f"{self.labels} of type {type(self.labels)}.\n"
            )


@dataclass(slots=True)
class Colors(ParserWord):
    """
    Literally colors to be used in plots.
    """

    colors: dict[Hashable, Color] | list[Color] | set[Color]
    name: Hashable = None

    def __len__(self):
        return len(self.colors)

    def could_refer_to(self, other):
        if isinstance(other, Numeric) and (
            (len(self) == 1) or (len(self) == len(other))
        ):
            return True
        else:
            return False


@dataclass(slots=True)
class Command(ParserWord):
    """
    These carry instructions about how other ParserWords should be interpreted.
    """

    command: CommandNames

    def could_refer_to(self, other):
        if (
            (
                (self.command == CommandNames.GLOBAL)
                and (
                    isinstance(other, Transformation)
                    or (isinstance(other, Colors) and (len(other) == 1))
                )
            )
            or (
                (
                    self.command
                    == CommandNames.IGNORE
                    # IGNORE must refer to labels, which then indicate what to ignore.
                )
                and isinstance(other, Labels)
            )
            or (
                (self.command == CommandNames.READ_AS_COLOR)
                and (
                    isinstance(other, (Numeric, Labels))
                    and isinstance(other.to_color_if_possible(), Colors)
                )
            )
        ):
            return True
        else:
            return False


@dataclass(slots=True)
class Transformation(ParserWord):
    """
    These can be used to modify Numeric items for visual representation. For
    example, a transformation may be desirable to make the radius of dots on a
    scatter plot proportional to some data, rather than the area (default), or
    arbitrary functions like [area = -log(x + 1)], etc. Axes, size legend, etc.
    Will be labeled using the un-transformed data values.
    """

    func: Callable

    def could_refer_to(self, other):
        return isinstance(other, Numeric)


def _process_data_frame(
    arg_index: int, data_frame: pd.DataFrame
) -> list[ParserWord]:

    parser_words = []
    for col_index, col_key in enumerate(data_frame.columns):
        if pd.api.types.is_numeric_dtype(data_frame[col_key].dtype):
            parser_words.append(
                Numeric(
                    arg_indices=DefaultList([arg_index, col_index]),
                    numbers=np.array(data_frame[col_key]),
                    name=col_key,
                )
            )
        else:
            parser_words.append(
                Labels(
                    arg_indices=DefaultList([arg_index, col_index]),
                    labels=list(data_frame[col_key]),
                    name=col_key,
                )
            )

    if type(data_frame.index) == Index:
        # The index of a DataFrame created from a dict without subsequently
        #   setting this will have a different type, and is ignored.
        parser_words.append(
            Labels(
                arg_indices=DefaultList([arg_index, len(data_frame.columns)]),
                labels=list(data_frame.index),
            )
        )

    return parser_words


def _process_series(arg_index: int, series: Series) -> list[ParserWord]:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return [
            Numeric(
                arg_indices=DefaultList([arg_index]),
                numbers=np.array(series),
                name=series.name,
            )
        ]
    else:
        return [
            Labels(
                arg_indices=DefaultList([arg_index]),
                labels=list(series),
                name=series.name,
            )
        ]


def _process_array(arg_index: int, array: NDArray) -> list[ParserWord]:
    return [
        Numeric(
            arg_indices=DefaultList([arg_index]),
            numbers=array,
        )
    ]


def _process_dict(arg_index: int, dictionary: dict) -> list[ParserWord]:
    if all_are_instances(dictionary.values(), (int, float)):
        return [
            Numeric(
                arg_indices=DefaultList([arg_index]),
                numbers=dictionary,
            )
        ]
    elif all_are_instances(dictionary.values(), Hashable):
        return [
            Labels(
                arg_indices=DefaultList([arg_index]),
                labels=dictionary,
            )
        ]
    elif all_are_instances(dictionary.values(), Color):
        return [
            Colors(
                arg_indices=DefaultList([arg_index]),
                labels=dictionary,
            )
        ]
    else:  # dict could be used to specify names for ParserWords
        parser_words = []
        for name, value in dictionary.items():
            new_words = _single_arg_to_parser_words(value, arg_index)
            for word in new_words:
                if hasattr(word, "name"):
                    word.name = name
                word.arg_indices = DefaultList([arg_index] + word.arg_indices)
                parser_words.append(word)
        return parser_words


def _process_list(arg_index: int, elements) -> list[ParserWord]:
    if all_are_instances(elements, (int, float)):
        return [
            Numeric(
                arg_indices=DefaultList([arg_index], -1),
                numbers=np.array(elements),
            )
        ]

    elif all_are_instances(elements, Hashable):
        if all((elem in CommandNames.__members__ for elem in elements)):
            return [
                Command(
                    arg_indices=DefaultList([arg_index]),
                    command=CommandNames[elem],
                )
                for elem in elements
            ]

        else:
            return [
                Labels(
                    arg_indices=DefaultList([arg_index]),
                    labels=elements,
                )
            ]


def _process_tuple(arg_index: int, tuple_arg) -> list[ParserWord]:
    if isinstance(tuple_arg, Color):
        return [
            Colors(
                arg_indices=DefaultList([arg_index]),
                colors={tuple_arg},
            )
        ]
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
        return [
            Command(
                arg_indices=DefaultList([arg_index]),
                command=CommandNames[string],
            )
        ]
    else:  # Interpreted as a single Label
        return [
            Labels(
                arg_indices=DefaultList([arg_index]),
                labels={string},
                name=string,
            )
        ]


def _process_callable(arg_index: int, callable: Callable) -> list[ParserWord]:
    try:
        return [
            Transformation(
                arg_indices=DefaultList([arg_index]),
                func=wrap_transformation_func(callable),
            )
        ]
    except Exception as e:
        warnings.warn(
            "Failed to create Transformation(ParserWord) from Callable at "
            + f"{arg_index=}. Exception: {str(e)}.\n"
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
                + " dict, string, Color, or list.\n"
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


parser_word_types_info = pd.DataFrame(
    {
        "class": [Numeric, Colors, Labels, Command, Transformation],
        "iterable": [True, True, True, False, False],
    }
)


class ParserWordTypes(Enum):
    NUMERIC = auto()
    COLORS = auto()
    LABELS = auto()
    COMMAND = auto()
    TRANSFORMATION = auto()


@dataclass
class ParsingDirective:
    """
    ParsingDirective objects essentially specify the data required (or that
    could optionally be used) to make a particular type of plot. For example,
    a scatter plot requires paired X and Y coordinate Numerics or equal length,
    and can additionally incorporate up to two more as size and color data, as
    well as axis labels and a title.
    """

    equal_length_required: list[ParserWordTypes] = []
    equal_length_optional: list[ParserWordTypes] = []
    individual_required: list[ParserWordTypes] = []
    individual_optional: list[ParserWordTypes] = []

    @classmethod
    def monotone_scatter(cls):
        return cls(
            equal_length_required=[
                ParserWordTypes.NUMERIC,  # x
                ParserWordTypes.NUMERIC,  # y
            ],
            equal_length_optional=[
                ParserWordTypes.NUMERIC,  # size (individual)
                ParserWordTypes.LABELS,  # point names
                ParserWordTypes.LABELS,  # markers
            ],
            individual_optional=[
                ParserWordTypes.NUMERIC,  # size (overall)
                ParserWordTypes.LABELS,  # title
                ParserWordTypes.LABELS,  # x label
                ParserWordTypes.LABELS,  # y label
            ],
        )


def _arg_parser(
    requirements: list[ParsingDirective], settings: SettingsPacket, *args
):
    """
    This function converts arguments from the user into a ParserWord list, then
    analyzes the "grammar" of that list to determine which items modify others,
    and what purposes they should fill in making a plot.

    "directive" provides context about what to search for (e.g. x, y, c)
    values for a color-map scatter plot
    """

    words: list[ParserWord] = _args_to_parser_words(*args)
    state: ParserState = ParserState.initial_state(settings)
    reference_matrix = np.zeros([len(words)] * 2, dtype=np.bool_)

    # First pass over words to build reference matrix
    last_arg_depth = 1
    state_backups = []
    for i, word in enumerate(words):
        # manage state context for potentially nested tuples of args
        this_arg_depth = len(word.arg_indices)
        if this_arg_depth > last_arg_depth:
            # when entering a tuple, back up state
            state_backups += [copy.deepcopy(state)] * (
                this_arg_depth - last_arg_depth
            )
        elif this_arg_depth < last_arg_depth:
            # when exiting a tuple, restore outside state
            for _ in range(last_arg_depth - this_arg_depth):
                state = state_backups.pop(-1)

        reference_matrix[i, :] = state.possible_referents_mask(words, i)
        if isinstance(word, Command):
            state.obey(word)

    # Reset state for second pass through words
    state = ParserState.initial_state(settings)

    last_arg_depth = 1
    state_backups = []
    for i, word in enumerate(words):
        # manage state context for potentially nested tuples of args
        this_arg_depth = len(word.arg_indices)
        if this_arg_depth > last_arg_depth:
            # when entering a tuple, back up state
            state_backups += [copy.deepcopy(state)] * (
                this_arg_depth - last_arg_depth
            )
        elif this_arg_depth < last_arg_depth:
            # when exiting a tuple, restore outside state
            for _ in range(last_arg_depth - this_arg_depth):
                state = state_backups.pop(-1)

        if isinstance(word, Command):
            state.obey(word)

        requirements.ingest(word)

        last_arg_depth = this_arg_depth


class ParserState:
    """
    Holds temporary internal state information of the parser.
    """

    def __init__(
        self,
        scope: str,
        assert_valid: bool = True,
    ):
        self.scope: str = scope
        self.pending_ignore_target: bool = False
        self.convert_to_color_if_possible: bool = False

        if assert_valid:
            self.assert_validity()

    def __eq__(self, other):
        if (
            isinstance(other, ParserState)
            and (self.scope == other.scope)
            and (self.pending_ignore_target == other.pending_ignore_target)
            and (
                self.convert_to_color_if_possible
                == other.convert_to_color_if_possible
            )
        ):
            return True
        else:
            return False

    @classmethod
    def initial_state(cls, settings):
        return cls(
            settings.default_parser_scope,
        )

    def assert_validity(self):
        assert self.scope in {"global", "next_arg", "next_hit"}
        assert isinstance(self.pending_ignore_target, bool)
        assert isinstance(self.convert_to_color_if_possible, bool)

    def obey(self, command):
        if isinstance(command, Command):
            match command.command:
                case CommandNames.GLOBAL:
                    self.scope = "global"
                case CommandNames.IGNORE:
                    self.pending_ignore_target = True
                case CommandNames.READ_AS_COLOR:
                    self.convert_to_color_if_possible = True
        else:
            raise TypeError(
                "ParserState can only obey a Command, not {command}.\n"
            )

    def possible_referents_mask(self, words, word_index):
        compatibility_mask = np.array(
            [words[word_index].could_refer_to(word) for word in words]
        )

        if self.scope == "global":
            # select all other words, at least within the tuple if this word
            # came form an element of a tuple within the args.
            if len(words[word_index].arg_indices) > 1:
                return (
                    (np.arange(len(words)) != word_index)
                    & np.array(
                        [
                            (
                                words[word_index].arg_indices[:-1]
                                == word.arg_indices[
                                    : len(words[word_index].arg_indices) - 1
                                ]
                            )
                            and (i != word_index)
                            for i, word in words
                        ]
                    )
                    & compatibility_mask
                )
            else:
                return (
                    np.arange(len(words)) != word_index
                ) & compatibility_mask

        elif self.scope == "next_arg":
            # Select words with the "next" arg_indices (slightly complicated,
            #   because tuple args may be treated as having sub-args).
            next_arg_indices = copy.deepcopy(words[word_index].arg_indices)
            next_arg_indices[-1] += 1
            return (
                np.array(
                    [
                        word.arg_indices[: len(next_arg_indices)]
                        == next_arg_indices
                        for word in words
                    ]
                )
                & compatibility_mask
            )

        elif self.scope == "next_hit":
            # Select words just the next compatible word.
            mask = np.zeros(len(words), dtype=np.bool_)
            for i in range(word_index, len(words)):
                if compatibility_mask[i]:
                    mask[i] = True
                    break
            return mask

        elif self.scope in possible_parser_scopes:
            raise InvalidParserState(
                f"ParserState.scope = {self.scope} present in "
                + "_input_parsing.possible_parser_scopes, but not implemented "
                + "in ParserState.possible_referents_mask() (oops)."
            )

        else:
            raise InvalidParserState(
                "ParserState.scope = {self.scope} is not valid. It must be in "
                + f"{possible_parser_scopes}."
            )
