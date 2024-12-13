"""
This module holds miscelaneous utilities that are used by other modules but do
not clearly fall withingh their scope.
"""

import warnings
from typing import Callable, Generator, Iterable
from collections.abc import Hashable
import numpy as np
from numpy.typing import NDArray


class PassthroughDict(dict):
    """
    Slightly modified dictionary that returns the key rather than raising an
    an error when you try to get an item with a key it doesn't have. This is
    useful for the "disp_names" dictionaries used in various plotter functions.
    It also requires both the key and value to be hashable, allowing the mapping
    to be inverted (swapping the keys with the values).
    """

    def __new__(cls, dictionary: dict):
        self = super().__new__(cls, dictionary)
        if not all(
            (isinstance(value, Hashable) for value in dictionary.values())
        ):
            unhashables = [
                f"{key} : {value}"
                for key, value in dictionary.items()
                if isinstance(dictionary[key], Hashable)
            ]
            raise TypeError(
                "PassthroughDict.__init__() recieved unhashable value(s) "
                + f"{unhashables}. Only hashable keys and values are allowed."
            )
        return self

    def __missing__(self, key: Hashable):
        # Return key instead of raising KeyError. This is what is meant by
        #   "Passthrough".
        return key

    @property
    def inverse(self):
        """
        PassthroughDict is used to translate short keys to formatted display
        names (and similar applications). This produces the object to translate
        back to the short keys.
        """
        return self.__class__({value: key for key, value in self.items()})

    def __eq__(self, other):
        if isinstance(other, PassthroughDict):
            if len(self) == len(other):
                return all(
                    (self_key == other_key) and (self_value == other_value)
                    for (self_key, self_value), (other_key, other_value) in zip(
                        self.items(), other.items()
                    )
                )
            else:
                return False
        else:
            return super().__eq__(other)


class DefaultList(list):
    """
    Slightly modified list class that allows "getting" a default value for
    indices outside the length of the actual list.
    """

    def __init__(self, normal_list, default_value=None):
        super().__init__(normal_list)
        self.default_value = default_value

    def __getitem__(self, index):
        if isinstance(index, int):
            if -len(self) <= index < len(self):
                return super().__getitem__(index)
            else:
                return self.default_value
        elif isinstance(index, slice):
            if index.step is not None:
                if index.start is not None:
                    return [
                        self[i]
                        for i in range(index.start, index.stop, index.step)
                    ]
                else:
                    return [self[i] for i in range(0, index.stop, index.step)]
            else:
                if index.start is not None:
                    return [self[i] for i in range(index.start, index.stop)]
                else:
                    return [self[i] for i in range(index.stop)]
        else:
            raise TypeError(
                f"DefaultList.__getitem__() got argument {index}, which is not "
                + "an integer or a slice, so it didn't work.\n"
            )


class InvalidTransformationFunc(Exception):
    def __init__(
        self, message="Transformation should map any float to another float."
    ):
        self.message = message


def is_nan_or_inf(array: NDArray) -> NDArray[np.bool_]:
    return np.isnan(array) | np.isinf(array)


def wrap_transformation_func(
    callable: Callable, test: bool = False
) -> Callable:
    """
    This wraps a function that could be used to transform a numpy array by
    making sure its input and output have dtype np.float64, and can also test
    the function to make sure it doesn't throw errors on nan or inf.
    """
    try:
        # Wrapped version forces input and output to be np.float64
        func = lambda x: np.array(
            callable(np.array(x, dtype=np.float64)), dtype=np.float64
        )

        if test:
            with np.errstate(invalid="ignore"):  # supresses log(-1) warning
                np.seterr(divide="ignore")  # supresses divide by zero warning
                test_array = np.array(
                    [np.nan, np.inf, -np.inf, -1, 0, 1, 10000]
                )
                result = func(test_array)

                assert isinstance(result, np.ndarray)
                assert result.shape[0] == test_array.shape[0]

        return func

    except Exception as e:
        raise InvalidTransformationFunc(
            "Callable could not be wrapped or raised error when tested.\n"
        ) from e


def all_are_instances(iterable, specified_type):
    """
    This is like isinstance, but to check all values of an iterable.
    """
    if isinstance(iterable, Iterable):
        if isinstance(iterable, dict):
            return all(
                isinstance(elem, specified_type) for elem in iterable.values()
            )
        return all(isinstance(elem, specified_type) for elem in iterable)
    else:
        raise ValueError(
            "all_are_instances accepts first arg of type list, generator,"
            + f" or tuple, not{type(iterable)}.\n"
        )


possible_parser_scopes = {"global", "next_arg", "next_hit"}
