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
        ScopeModes,
        wrap_transformation_func,
        all_are_instances,
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
            ScopeModes,
            wrap_transformation_func,
            all_are_instances,
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

import logging
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

    # Scope modifiers:
    GLOBAL = auto()
    NEXT_ARG = auto()
    NEXT_WORD = auto()
    NEXT_HIT = auto()
    # read instructions
    READ_AS_COLOR = auto()  # actually implemented as conversion on second pass
    READ_AS_MATRIX = auto()
    # other
    IGNORE = auto()


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
                + "isinstance(self.arg_indices, DefaultList) = "
                + f"{isinstance(self.arg_indices, DefaultList)}\n"
                + f"{self.arg_indices=}\n"
                + f"arg_indices_types: {[type(i) for i in self.arg_indices]}"
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
class Matrix(ParserWord):
    """
    Matrix data, e.g. such as image/heatmap values.
    """

    numbers: NDArray[np.floating | np.integer]
    axis_0_labels: list[Hashable] = None
    axis_1_labels: list[Hashable] = None
    name: Hashable | None = None
    assert_valid: bool = True

    def assert_validity(self):
        if (
            ParserWord.assert_validity(self)
            and isinstance(self.name, Hashable)
            and isinstance(self.numbers, (dict, np.ndarray, set))
            and isinstance(self.assert_valid, bool)
            and (
                np.issubdtype(self.numbers.dtype, np.floating)
                or np.issubdtype(self.numbers.dtype, np.integer)
            )
            and (self.numbers.shape[0] == len(self.axis_0_labels))
            and (self.numbers.shape[1] == len(self.axis_1_labels))
        ):
            return True
        elif self.assert_valid is False:
            warnings.warn(
                "Attempted to assert validity of an invalid Matrix(ParserWord)"
                + ". This did not raise an error because it also had attribute"
                + " assert_valid=False.\n"
            )
        else:
            raise MalformedParserWord(
                f"{str(self)} is not a valid Matrix(ParserWord)."
            )

    def __post_init__(self):
        if not self.axis_0_labels:
            self.axis_0_labels = list(range(self.numbers.shape[0]))
            self.axis_1_labels = list(range(self.numbers.shape[1]))

        ParserWord.__post_init__(self)

        if self.assert_valid:
            self.assert_validity()

    @property
    def shape(self):
        return self.numbers.shape


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
            (hasattr(other, "name") and (other.name in self.labels))
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

    @classmethod
    def from_name(cls, arg_indices, command_name):
        if isinstance(command_name, CommandNames):
            return cls(arg_indices, command_name)

        elif command_name in CommandNames.__members__:
            return cls(arg_indices, CommandNames[command_name])

    def could_refer_to(self, other):
        # note that READ_AS_MATRIX must return false, because it refers to args
        #   not ParserWords. This is not how READ_AS_COLOR works.
        if (
            (
                (
                    (self.command == CommandNames.GLOBAL)
                    or (self.command == CommandNames.NEXT_ARG)
                    or (self.command == CommandNames.NEXT_WORD)
                    or (self.command == CommandNames.NEXT_HIT)
                )
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


class ParserInternalHitCodes(Enum):
    IGNORED = -1
    READ_COLOR = -2
    READ_MATRIX = -3
    USED_SCOPE = -4


@dataclass
class ParserState:
    """
    Holds temporary internal state information of the parser.
    """

    default_scope: ScopeModes = ScopeModes.GLOBAL
    ignoring_is_default: bool = False
    convert_to_color_is_default: bool = False
    read_as_matrix_is_default: bool = False
    assert_valid: bool = True

    def __post_init__(self):
        # _stack attributes are lists of tuples, where the first element of the
        #   last tuple determines the actual scope, ignoring, etc. status. The
        #   second element of each tuple is a ScopeMode, which determines when
        #   that tuple might be removed form the list.

        self._scope_stack: list[tuple[ScopeModes, ScopeModes]] = [
            (self.default_scope, ScopeModes.GLOBAL)
        ]
        self._ignoring_stack: list[tuple[bool, ScopeModes]] = [
            (self.ignoring_is_default, ScopeModes.GLOBAL)
        ]
        self._convert_to_color_stack: list[tuple[bool, ScopeModes]] = [
            (self.convert_to_color_is_default, ScopeModes.GLOBAL)
        ]
        self._read_as_matrix_stack: list[tuple[bool, ScopeModes]] = [
            (self.read_as_matrix_is_default, ScopeModes.GLOBAL)
        ]
        self._indices_pending_hit: set[int] = set()
        self.current_word_index = 0

        if self.assert_valid:
            self.assert_validity()

    @property
    def scope(self):
        return self._scope_stack[-1][0]

    @scope.setter
    def scope(self, value: ScopeModes):
        assert isinstance(value, ScopeModes)
        return self._scope_stack.append((value, self.scope))

    @property
    def ignoring(self):
        return self._ignoring_stack[-1][0]

    @ignoring.setter
    def ignoring(self, value: bool):
        assert isinstance(value, bool)
        return self._ignoring_stack.append((value, self.scope))

    @property
    def convert_to_color(self):
        return self._convert_to_color_stack[-1][0]

    @convert_to_color.setter
    def convert_to_color(self, value: bool):
        assert isinstance(value, bool)
        return self._convert_to_color_stack.append((value, self.scope))

    @property
    def read_as_matrix(self):
        return self._read_as_matrix_stack[-1][0]

    @read_as_matrix.setter
    def read_as_matrix(self, value: bool):
        assert isinstance(value, bool)
        return self._read_as_matrix_stack.append((value, self.scope))

    def __eq__(self, other):
        """
        note: this will return true if the two parser states are momentarily,
            functioanlly identical; _stack differences are not captured.
        """
        if (
            isinstance(other, ParserState)
            and (self.scope == other.scope)
            and (self.ignoring == other.ignoring)
            and (self.convert_to_color == other.convert_to_color)
            and (self.read_as_matrix == other.read_as_matrix)
        ):
            return True
        else:
            return False

    @classmethod
    def initial_state(cls, settings):
        return cls(
            default_scope=settings.parser_default_scope,
            ignoring_is_default=settings.parser_default_ignoring,
            convert_to_color_is_default=settings.parser_default_read_as_color,
            read_as_matrix_is_default=settings.parser_default_read_as_matrix,
            assert_valid=settings.parser_default_assert_valid,
        )

    def assert_validity(self):
        assert self.scope in ScopeModes.__members__.values()
        assert isinstance(self.ignoring, bool)
        assert isinstance(self.convert_to_color, bool)
        assert isinstance(self.read_as_matrix, bool)
        # for _stack

    def _stack_tuple_stuff(self):
        if self.scope is ScopeModes.NEXT_HIT:
            return (self.scope, self.current_word_index)
        else:
            return (self.scope,)

    def obey(self, command):
        if isinstance(command, Command):
            match command.command:
                case CommandNames.GLOBAL:
                    self._scope_stack.append(
                        (ScopeModes.GLOBAL, *self._stack_tuple_stuff())
                    )
                case CommandNames.NEXT_WORD:
                    self._scope_stack.append(
                        (ScopeModes.NEXT_WORD, *self._stack_tuple_stuff())
                    )
                case CommandNames.NEXT_ARG:
                    self._scope_stack.append(
                        (ScopeModes.NEXT_ARG, *self._stack_tuple_stuff())
                    )
                case CommandNames.NEXT_HIT:
                    self._scope_stack.append(
                        (ScopeModes.NEXT_HIT, *self._stack_tuple_stuff())
                    )
                case CommandNames.IGNORE:
                    self._ignoring_stack.append(
                        (True, *self._stack_tuple_stuff())
                    )
                case CommandNames.READ_AS_COLOR:
                    self._convert_to_color_stack.append(
                        (True, *self._stack_tuple_stuff())
                    )
                case CommandNames.READ_AS_MATRIX:
                    self._read_as_matrix_stack.append(
                        (True, *self._stack_tuple_stuff())
                    )
                case _:
                    raise MalformedParserWord(
                        "ParserState.obey() recieved unrecognized command "
                        + str(command.command)
                    )
            self.hit_tick(ParserInternalHitCodes.USED_SCOPE)
        else:
            raise TypeError(
                "ParserState can only obey a Command, not {command}.\n"
            )

    def possible_referents_mask(self, words, word_index):
        compatibility_mask = np.array(
            [words[word_index].could_refer_to(word) for word in words]
        )

        if self.scope in {
            ScopeModes.GLOBAL,
            ScopeModes.NEXT_WORD,
            ScopeModes.NEXT_HIT,
        }:
            # select all other words, at least within the tuple if this word
            # came from an element of a tuple within the args.

            if len(words[word_index].arg_indices) > 1:
                return (
                    (np.arange(len(words)) > word_index)
                    & np.array(
                        [
                            (
                                words[word_index].arg_indices[:-1]
                                == word.arg_indices[
                                    : len(words[word_index].arg_indices) - 1
                                ]
                            )
                            and (i != word_index)
                            for i, word in enumerate(words)
                        ]
                    )
                    & compatibility_mask
                )
            else:
                return (
                    np.arange(len(words)) != word_index
                ) & compatibility_mask

        elif self.scope == ScopeModes.NEXT_ARG:
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

        elif self.scope in ScopeModes.__members__.values():
            raise InvalidParserState(
                f"ParserState.scope = {self.scope} present in "
                + "_utils.ScopeModes.__members__, but not implemented "
                + "in ParserState.possible_referents_mask() (oops)."
            )

        else:
            raise InvalidParserState(
                "ParserState.scope = {self.scope} is not "
                + f"valid. It must be in {ScopeModes.__members__}.\n"
            )

    def arg_tick(self):
        self.word_tick()
        for _stack in [
            self._convert_to_color_stack,
            self._ignoring_stack,
            self._read_as_matrix_stack,
            self._scope_stack,
        ]:
            _stack = [
                item for item in _stack if not item[1] is ScopeModes.NEXT_ARG
            ]

    def word_tick(self):
        for _stack in [
            self._convert_to_color_stack,
            self._ignoring_stack,
            self._read_as_matrix_stack,
            self._scope_stack,
        ]:
            _stack = [
                item for item in _stack if not item[1] is ScopeModes.NEXT_WORD
            ]

        self.current_word_index += 1

    def hit_tick(self, hitter_index):
        if hitter_index in self._indices_pending_hit:
            self._indices_pending_hit.remove(hitter_index)

            for _stack in [
                self._convert_to_color_stack,
                self._ignoring_stack,
                self._read_as_matrix_stack,
                self._scope_stack,
            ]:
                _stack = [
                    item
                    for item in _stack
                    if not (
                        (item[1] is ScopeModes.NEXT_HIT)
                        and (item[2] == hitter_index)
                    )
                ]


def _process_data_frame(
    arg_index: int,
    data_frame: pd.DataFrame,
    parser_state: ParserState,
) -> list[ParserWord]:
    potential_matrix = all(
        pd.api.types.is_numeric_dtype(data_frame[col])
        for col in data_frame.columns
    )
    if parser_state.read_as_matrix and potential_matrix:
        parser_state.hit_tick(ParserInternalHitCodes.READ_MATRIX)
        return [
            Matrix(
                arg_indices=DefaultList([arg_index]),
                numbers=np.array(data_frame),
                axis_0_labels=list(data_frame.index),
                axis_1_labels=list(data_frame.columns),
            )
        ]
    else:
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
                    arg_indices=DefaultList(
                        [arg_index, len(data_frame.columns)]
                    ),
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


def _process_array(
    arg_index: int,
    array: NDArray,
    parser_state: ParserState,
) -> list[ParserWord]:
    if parser_state.read_as_matrix or (len(array.shape) == 2):
        parser_state.hit_tick(ParserInternalHitCodes.READ_MATRIX)
        return [
            Matrix(
                arg_indices=DefaultList([arg_index]),
                numbers=array,
            )
        ]
    elif len(array.shape) == 1:
        return [
            Numeric(
                arg_indices=DefaultList([arg_index]),
                numbers=array,
            )
        ]
    else:
        warnings.warn(
            f"{len(array.shape)}-dimensional array flattened to Numeric."
        )
        return [
            Numeric(
                arg_indices=DefaultList([arg_index]),
                numbers=array.flatten(),
            )
        ]


def _process_dict(
    arg_index: int,
    dictionary: dict,
    parser_state: ParserState,
    settings_packet: SettingsPacket,
) -> list[ParserWord]:
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
            new_words, parser_state = _single_arg_to_parser_words(
                value,
                arg_index,
                parser_state,
                settings_packet,
            )
            for word in new_words:
                if hasattr(word, "name"):
                    word.name = name
                word.arg_indices = DefaultList([arg_index] + word.arg_indices)
                parser_words.append(word)
        return parser_words


def _process_list(
    arg_index: int,
    elements: list,
    parser_state: ParserState,
    settings_packet: SettingsPacket,
) -> list[ParserWord]:
    if all_are_instances(elements, (int, float)):
        return [
            Numeric(
                arg_indices=DefaultList([arg_index], -1),
                numbers=np.array(elements),
            )
        ]

    elif all_are_instances(elements, Hashable):
        if all(elem in CommandNames.__members__ for elem in elements):
            return _process_tuple(
                arg_index,
                tuple(elements),
                parser_state,
                settings_packet,
            )

        else:
            return [
                Labels(
                    arg_indices=DefaultList([arg_index]),
                    labels=elements,
                )
            ]
    else:
        return _process_tuple(
            arg_index,
            tuple(elements),
            parser_state,
            settings_packet,
        )


def _process_tuple(
    arg_index: int,
    tuple_arg: tuple,
    parser_state: ParserState,
    settings_packet: SettingsPacket,
) -> list[ParserWord]:
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
        parser_state_backup = copy.deepcopy(parser_state)
        parser_words = _args_to_parser_words(
            parser_state, settings_packet, *tuple_arg
        )
        for elem in parser_words:
            # prepend outer index
            elem.arg_indices.insert(0, arg_index)
        parser_state = parser_state_backup
        return parser_words


def _process_string(arg_index: int, string: str) -> list[ParserWord]:
    # note: a string is never interpreted as a color at this point, but if this
    #   produces a Lables object with a string that could represent a color, it
    #   should be converted to a Color if referenced by a READ_AS_COLOR command.
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


def _single_arg_to_parser_words(
    arg,
    arg_index: int,
    parser_state: ParserState,
    settings_packet: SettingsPacket,
) -> tuple[list[ParserWord], ParserState]:
    """
    This function does most of the work for _args_to_parser_words() (see that
    function's docstring for more explanation) and is also used directly by
    _process_dict(). It takes something passed to plotomata by a user and routes
    it to one of several function (depending on type) for conversion to
    list[ParserWord] format.
    """

    match arg:
        case pd.DataFrame():
            new_words = _process_data_frame(arg_index, arg, parser_state)

        case Series():
            new_words = _process_series(arg_index, arg)

        case np.ndarray():
            new_words = _process_array(arg_index, arg, parser_state)

        case list():
            new_words = _process_list(
                arg_index, arg, parser_state, settings_packet
            )

        case dict():
            new_words = _process_dict(
                arg_index, arg, parser_state, settings_packet
            )

        case tuple():
            new_words = _process_tuple(
                arg_index,
                arg,
                parser_state,
                settings_packet,
            )

        case str():
            new_words = _process_string(arg_index, arg)
            if (len(new_words) == 1) and isinstance(new_words[0], Command):
                parser_state.obey(new_words[0])
                if new_words[0].command == CommandNames.IGNORE:
                    # The ignore command changes the parser state to prevent
                    #   subsequent words from being read on the first pass. It
                    #   must also effectively prevent itself being read, to
                    #   avoid having an unrelated effect on the second pass.
                    new_words = []

        case ParserWord():
            arg.arg_index = arg_index
            new_words = [arg]
            if isinstance(arg, Command):
                parser_state.obey(arg)

        case _ if callable(arg):
            new_words = _process_callable(arg_index, arg)

        case _:
            settings_packet.logger.warning(
                f"Unable to standardize argument {arg} of type {type(arg)}."
                + " Consider converting to DataFrame, Series, NDArray,"
                + " dict, string, Color, or list.\n"
            )
            new_words = []

    return new_words, parser_state


def _args_to_parser_words(
    parser_state, settings_packet: SettingsPacket, *args
) -> list[ParserWord]:
    """
    This function takes the args passed to a plotting function, which may come
    in a variety of types, determines what they could be used for (e.g. a
    series of numbers, a colormap, display names, etc.), and produces a new list
    of the same information in the form of ParserWords. Each arg could produce
    several control items or none.
    """
    settings_packet.logger.debug(
        f"_args_to_parser_words() got {len(args)} args."
    )
    parser_words: list[ParserWord] = []
    # indices_pending_hit: list[int] = []
    for arg_index, arg in enumerate(args):
        if parser_state.ignoring:
            settings_packet.logger.debug(f"ignored arg {arg}.")
            parser_state.hit_tick(ParserInternalHitCodes.IGNORED)
            new_parser_words = []
        else:
            new_parser_words, parser_state = _single_arg_to_parser_words(
                arg,
                arg_index,
                parser_state,
                settings_packet,
            )

        for pw in new_parser_words:
            if isinstance(pw, Command):
                parser_state.obey(pw)
            parser_state.word_tick()
            # for word_index in indices_pending_hit:
            #    if parser_words[word_index].could_refer_to(pw):
            #        # This currently doesn't do anything,
            #        indices_pending_hit.remove(word_index)
            # if parser_state.scope == ScopeModes.NEXT_HIT:
            #    indices_pending_hit.append(len(parser_words))

            parser_words.append(pw)

        parser_state.arg_tick()
    return parser_words


class ParserWordTypes(Enum):
    NUMERIC = auto()
    MATRIX = auto()
    COLORS = auto()
    LABELS = auto()
    COMMAND = auto()
    TRANSFORMATION = auto()

    def to_class(self):
        if self is ParserWordTypes.NUMERIC:
            return Numeric
        elif self is ParserWordTypes.MATRIX:
            return Matrix
        elif self is ParserWordTypes.COLORS:
            return Colors
        elif self is ParserWordTypes.LABELS:
            return Labels
        elif self is ParserWordTypes.COMMAND:
            return Command
        elif self is ParserWordTypes.TRANSFORMATION:
            return Transformation


parser_word_types_info = pd.DataFrame(
    {
        "class": [Numeric, Matrix, Colors, Labels, Command, Transformation],
        "type_enum": [
            ParserWordTypes.NUMERIC,
            ParserWordTypes.MATRIX,
            ParserWordTypes.COLORS,
            ParserWordTypes.LABELS,
            ParserWordTypes.COMMAND,
            ParserWordTypes.TRANSFORMATION,
        ],
        "iterable": [True, False, True, True, False, False],
    }
)


@dataclass
class PlotDataRequirements:
    """
    PlotDataRequirement objects essentially specify the data required (or that
    could optionally be used) to make a particular type of plot. For example,
    a scatter plot requires paired X and Y coordinate Numerics or equal length,
    and can additionally incorporate up to two more as size and color data, as
    well as axis labels and a title.
    """

    equal_length_required: dict[str, ParserWordTypes] = None
    equal_length_optional: dict[str, ParserWordTypes] = None
    individual_required: dict[str, ParserWordTypes] = None
    individual_optional: dict[str, ParserWordTypes] = None

    def __post_init__(self):
        if (self.equal_length_required is None) or (
            self.equal_length_required is False
        ):
            self.equal_length_required = {}

        if (self.equal_length_optional is None) or (
            self.equal_length_optional is False
        ):
            self.equal_length_optional = {}

        if (self.individual_required is None) or (
            self.individual_required is False
        ):
            self.individual_required = {}

        if (self.individual_optional is None) or (
            self.individual_optional is False
        ):
            self.individual_optional = {}

        self.assert_validity()

    def check_fulfillment(
        self,
        data: dict[str, ParserWord],
    ):
        equal_length_required_candidates: dict[str, ParserWord] = {
            key: data[key] for key in self.equal_length_required
        }
        equal_length_optional_candidates: dict[str, ParserWord] = {
            key: value
            for key, value in data.items()
            if key in self.equal_length_optional
        }
        individual_required_candidates: dict[str, ParserWord] = {
            key: data[key] for key in self.individual_required
        }
        individual_optional_candidates: dict[str, ParserWord] = {
            key: value
            for key, value in data.items()
            if key in self.individual_optional
        }

        for key in data:
            # no unrecognized keys allowed
            if not (
                (key in self.equal_length_optional)
                | (key in self.equal_length_required)
                | (key in self.individual_optional)
                | (key in self.individual_required)
            ):
                return False

        if len(equal_length_required_candidates) > 0:
            correct_length = len(
                list(equal_length_required_candidates.values())[0]
            )
        elif len(equal_length_optional_candidates) > 0:
            correct_length = len(
                list(equal_length_optional_candidates.values())[0]
            )
        else:
            logging.info(
                "PlotDataRequirement.check_fulfillment recieved no equal-length "
                + "candidates."
            )
            correct_length = -1

        return (
            # all arguments really are dict[str, ParserWord]
            isinstance(equal_length_required_candidates, dict)
            and isinstance(equal_length_optional_candidates, dict)
            and isinstance(individual_required_candidates, dict)
            and isinstance(individual_optional_candidates, dict)
            and all_are_instances(equal_length_required_candidates.keys(), str)
            and all_are_instances(equal_length_optional_candidates.keys(), str)
            and all_are_instances(individual_required_candidates.keys(), str)
            and all_are_instances(individual_optional_candidates.keys(), str)
            and all_are_instances(
                equal_length_required_candidates.values(), ParserWord
            )
            and all_are_instances(
                equal_length_optional_candidates.values(), ParserWord
            )
            and all_are_instances(
                individual_required_candidates.values(), ParserWord
            )
            and all_are_instances(
                individual_optional_candidates.values(), ParserWord
            )
            # all of the ParserWords have acceptable lengths
            and all(
                len(pw) == 1 for pw in individual_required_candidates.values()
            )
            and all(
                len(pw) == 1 for pw in individual_optional_candidates.values()
            )
            and all(
                len(pw) == correct_length
                for pw in equal_length_required_candidates.values()
            )
            and all(
                len(pw) == correct_length
                for pw in equal_length_optional_candidates.values()
            )
            # all of the required dictionary keys are present
            and all(
                key in individual_required_candidates
                for key in self.individual_required
            )
            and all(
                key in equal_length_required_candidates
                for key in self.equal_length_required
            )
            # all of the provided keys are present in self
            and all(
                key in self.individual_required
                for key in individual_required_candidates
            )
            and all(
                key in self.equal_length_required
                for key in equal_length_required_candidates
            )
            and all(
                key in self.individual_optional
                for key in individual_optional_candidates
            )
            and all(
                key in self.equal_length_optional
                for key in equal_length_optional_candidates
            )
            # all of the provided ParserWords have correct types
            and all(
                isinstance(
                    value,
                    parser_word_types_info["class"][
                        list(parser_word_types_info["type_enum"]).index(
                            self.individual_required[key]
                        )
                    ],
                )
                for key, value in individual_required_candidates.items()
            )
            and all(
                isinstance(
                    value,
                    parser_word_types_info["class"][
                        list(parser_word_types_info["type_enum"]).index(
                            self.individual_optional[key]
                        )
                    ],
                )
                for key, value in individual_optional_candidates.items()
            )
            and all(
                isinstance(
                    value,
                    parser_word_types_info["class"][
                        list(parser_word_types_info["type_enum"]).index(
                            self.equal_length_required[key]
                        )
                    ],
                )
                for key, value in equal_length_required_candidates.items()
            )
            and all(
                isinstance(
                    value,
                    parser_word_types_info["class"][
                        list(parser_word_types_info["type_enum"]).index(
                            self.equal_length_optional[key]
                        )
                    ],
                )
                for key, value in equal_length_optional_candidates.items()
            )
        )

    def assert_validity(self):
        for pwt_dict in [
            self.equal_length_optional,
            self.equal_length_required,
            self.individual_required,
            self.individual_optional,
        ]:
            assert isinstance(pwt_dict, dict)
            assert all_are_instances(pwt_dict.values(), ParserWordTypes)
            assert all_are_instances(pwt_dict.keys(), Hashable)

        assert all(  # not all ParserWordTypes make sense for equal length
            parser_word_types_info["class"][
                list(parser_word_types_info["type_enum"]).index(pwt)
            ]
            for pwt in list(self.equal_length_optional.values())
            + list(self.equal_length_required.values())
        )

        # make sure none of the ParserWord dicts have matching keys
        assert len(
            self.equal_length_optional
            | self.equal_length_required
            | self.individual_optional
            | self.individual_required
        ) == (
            len(self.equal_length_optional)
            + len(self.equal_length_required)
            + len(self.individual_optional)
            + len(self.individual_required)
        )

    @classmethod
    def monochrome_scatter(cls):
        return cls(
            equal_length_required={
                "x": ParserWordTypes.NUMERIC,
                "y": ParserWordTypes.NUMERIC,
            },
            equal_length_optional={
                "size_data": ParserWordTypes.NUMERIC,
                "point_names": ParserWordTypes.LABELS,
                "marker_types": ParserWordTypes.LABELS,
            },
            individual_optional={
                "size_transformation": ParserWordTypes.TRANSFORMATION,
                "size_scale": ParserWordTypes.NUMERIC,
                "title": ParserWordTypes.LABELS,
                "x_label": ParserWordTypes.LABELS,
                "y_label": ParserWordTypes.LABELS,
                "size_label": ParserWordTypes.LABELS,
            },
        )


def _arg_parser(
    requirements: list[PlotDataRequirements],
    settings: SettingsPacket,
    *args,
):
    """
    This function converts arguments from the user into a ParserWord list, then
    analyzes the "grammar" of that list to determine which items modify others,
    and what purposes they should fill in making a plot.

    requirements provides context about what to search for (e.g. x, y, c)
    values for a color-map scatter plot
    """

    # First pass over args to get words list
    state: ParserState = ParserState.initial_state(settings)
    words: list[ParserWord] = _args_to_parser_words(state, settings, *args)

    numeric_lengths: list[int] = [
        len(word) for word in words if isinstance(word, Numeric)
    ]

    labels_lengths: list[int] = [
        len(word) for word in words if isinstance(word, Labels)
    ]

    colors_lengths: list[int] = [
        len(word) for word in words if isinstance(word, Colors)
    ]

    word_lengths_set: set[int] = (
        set(numeric_lengths) | set(labels_lengths) | set(colors_lengths)
    )

    # Second pass over words to build reference matrix and length_scores
    reference_matrix = np.zeros([len(words)] * 2, dtype=np.bool_)
    state = ParserState.initial_state(settings)
    # last_arg_depth = 1
    last_arg_indices = [-1]
    state_backups = []
    for i, word in enumerate(words):
        state.current_word_index = i  # redundant with an effect of word_tick()

        last_arg_depth = len(last_arg_indices)
        if state.ignoring:
            state.hit_tick(ParserInternalHitCodes.IGNORED)
        else:
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

        state.word_tick()
        if not last_arg_indices == word.arg_indices:
            state.arg_tick()
            last_arg_indices = word.arg_indices

    for plot_req in requirements:
        n_eq_len_numerics_needed = np.sum(
            [
                pw_type is ParserWordTypes.NUMERIC
                for pw_type in plot_req.equal_length_required
            ]
        )
        n_eq_len_labels_needed = np.sum(
            [
                pw_type is ParserWordTypes.LABELS
                for pw_type in plot_req.equal_length_required
            ]
        )
        n_eq_len_colors_needed = np.sum(
            [
                pw_type is ParserWordTypes.COLORS
                for pw_type in plot_req.equal_length_required
            ]
        )
        n_eq_len_numerics_optional = np.sum(
            [
                pw_type is ParserWordTypes.NUMERIC
                for pw_type in plot_req.equal_length_optional
            ]
        )
        n_eq_len_labels_optional = np.sum(
            [
                pw_type is ParserWordTypes.LABELS
                for pw_type in plot_req.equal_length_optional
            ]
        )
        n_eq_len_colors_optional = np.sum(
            [
                pw_type is ParserWordTypes.COLORS
                for pw_type in plot_req.equal_length_optional
            ]
        )
        best_length: int = -1  # If a viable length is found, this will change.
        best_score: int = -1  # The length with the highest score will be used.
        for candidate_length in word_lengths_set:
            if (candidate_length == 1) and not settings.allow_one_point_plots:
                continue
            n_eq_len_numerics_available = np.sum(
                [
                    numeric_length == candidate_length
                    for numeric_length in numeric_lengths
                ]
            )
            n_eq_len_labels_available = np.sum(
                [
                    labels_length == candidate_length
                    for labels_length in labels_lengths
                ]
            )
            n_eq_len_colors_available = np.sum(
                [
                    colors_length == candidate_length
                    for colors_length in colors_lengths
                ]
            )
            if (
                (n_eq_len_numerics_available >= n_eq_len_numerics_needed)
                and (n_eq_len_labels_available >= n_eq_len_labels_needed)
                and (n_eq_len_colors_available >= n_eq_len_colors_needed)
            ):
                # there is enough data of length candidate_length to continue
                score = (
                    n_eq_len_numerics_available
                    - n_eq_len_numerics_needed
                    + n_eq_len_labels_available
                    - n_eq_len_labels_needed
                    + n_eq_len_colors_available
                    - n_eq_len_colors_needed
                    + np.log(candidate_length) * settings.prefer_longer_data
                )
                if score > best_score:
                    best_score = score
                    best_length = candidate_length

        if best_score >= 0:
            # There must be at least one viable candidate.
            data_for_plot = {}
            for word in words:
                word_used = False
                if len(word) == best_length:
                    for key, value in plot_req.equal_length_required.items():
                        if (key not in data_for_plot) and isinstance(
                            word, value.to_class()
                        ):
                            data_for_plot[key] = word
                            word_used = True
                            break

                if (len(word) == 1) and not word_used:
                    for key, value in plot_req.individual_required.items():
                        if (key not in data_for_plot) and isinstance(
                            word, value.to_class()
                        ):
                            data_for_plot[key] = word
                            word_used = True
                            break

                if (len(word) == best_length) and not word_used:
                    for key, value in plot_req.equal_length_optional.items():
                        if (key not in data_for_plot) and isinstance(
                            word, value.to_class()
                        ):
                            data_for_plot[key] = word
                            word_used = True
                            break

                if (len(word) == 1) and not word_used:
                    for key, value in plot_req.individual_optional.items():
                        if (key not in data_for_plot) and isinstance(
                            word, value.to_class()
                        ):
                            data_for_plot[key] = word
                            word_used = True
                            break

        else:
            settings.logger.warning(
                f"Data provided could not satisfy requirements {plot_req}.\n"
            )

    # Third pass over words with reference matrix guided by directive
    state = ParserState.initial_state(settings)
    last_arg_indices = [-1]
    state_backups = []
    for i, word in enumerate(words):
        this_arg_depth = len(word.arg_indices)
        last_arg_depth = len(last_arg_indices)
        if state.ignoring:
            state.hit_tick(ParserInternalHitCodes.IGNORED)
        else:
            # manage state context for potentially nested tuples of args
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

        state.word_tick()
        if not word.arg_indices == last_arg_indices:
            state.arg_tick()
        last_arg_indices = word.arg_indices
