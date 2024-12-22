import pytest
import numpy as np
from plotomata._utils import (
    PassthroughDict,
    wrap_transformation_func,
    InvalidTransformationFunc,
    all_are_instances,
    is_nan_or_inf,
    DefaultList,
    is_tuple_as_string,
)


@pytest.fixture
def capitalize_abc():
    return PassthroughDict(
        {
            "a": "A",
            "b": "B",
            "c": "C",
        }
    )


def test_PassthroughDict_inversion(capitalize_abc):
    correct_inverse = PassthroughDict(
        {
            "A": "a",
            "B": "b",
            "C": "c",
        }
    )
    assert capitalize_abc.inverse == correct_inverse


def test_PassthroughDict_passthrough(capitalize_abc):
    assert capitalize_abc["d"] == "d"


def test_PassthroughDict_validity():
    with pytest.raises(TypeError):
        _ = PassthroughDict(
            {
                "a": [],  # cannot create a PassthroughDict with non-hashable value.
            }
        )


def test_wrap_transformation_func_exception():
    with pytest.raises(InvalidTransformationFunc):
        _ = wrap_transformation_func(lambda x: None, test=True)


def test_wrap_transformation_func_works():
    func = wrap_transformation_func(lambda x: np.log(x), test=True)

    test_return = func(np.ones(2)) - np.zeros(2, dtype=np.float64)
    assert np.all(np.abs(test_return) < 0.001)


def test_all_are_instances():
    assert all_are_instances([2 for _ in range(1)], int) is True
    assert all_are_instances((2 for _ in range(1)), int) is True
    assert all_are_instances((2,), int) is True
    assert all_are_instances((2.0,), int) is False
    assert all_are_instances({1: "one"}, str)


def test_is_nan_or_inf():
    assert np.all(
        is_nan_or_inf(np.array([0, -np.inf, np.nan]))
        == np.array([False, True, True])
    )


def test_DefaultList():
    dl = DefaultList([1, 2], default_value=3)
    assert dl[2] == 3
    assert dl[1:4] == [2, 3, 3]
    assert dl[-2] == 1
    assert dl[-3] == 3
    assert dl[-4:-1] == [3, 3, 1]


def test_is_tuple_as_string():
    assert is_tuple_as_string("(1, 2, '3', '$%^&*()')")
    assert is_tuple_as_string("(1, 2, '3', '$%^&*()', )") is False
    assert is_tuple_as_string("()")
    assert is_tuple_as_string("(__, ZHBUHjbm, 2323)")
    assert is_tuple_as_string("1, Z, __") is False
