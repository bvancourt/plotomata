import warnings
import pytest
import pandas as pd
import numpy as np
from plotomata._input_parsing import (
    Numbers, 
    Labels,
    Colors,
    ParsingCommand,
    Transformation,
    CommandNames,
    _args_to_parser_words,
    _process_array,
    _process_callable,
    _process_data_frame,
    _process_dict,
    _process_list,
    _process_series,
    _process_string,
    _process_tuple,
)
from plotomata._utils import PassthroughDict
from plotomata.color_palettes import Color

@pytest.fixture
def numbers_pw():
    return Numbers(
        arg_indices = [0],
        numbers = np.array([1,2,3]),
        name = "counting",
    )

@pytest.fixture
def test_dict():
    return {"x":[1,2,3], "y":[4,2,3], "z":["five","seven","six"]}

@pytest.fixture
def test_data_frame(test_dict):
    return pd.DataFrame(test_dict)

@pytest.fixture
def junk_tuple(test_dict, test_data_frame):
    return (
        test_data_frame, # produces 3 p.w.
        test_dict["x"],
        PassthroughDict({"x":7, "y":7, "z":7}),
        np.array([6,6,6,7]),
        ["a", "b", "c", "d"],
        Color((1,0,1)),
        (np.array([6,6,6,7]), ["a", "b", "c", "d"]), # produces 2 ParserWords
        lambda a: np.sqrt(a),
        "IGNORE",
    )

def test_Numbers_to_Colors_conversion(numbers_pw):
    colors_pw = numbers_pw.to_color_if_possible()
    assert isinstance(colors_pw, Colors)

def test_Labels_init():
    labels = Labels(
        arg_indices = [0],
        labels = [1,None,"three"],
        name = "counting",
    )
    assert isinstance(labels, Labels)

def test_Colors_init():
    colors = Colors(
        arg_indices = [0],
        colors = [Color(1,2,3)],
        name = "dark",
    )
    assert isinstance(colors, Colors)

def test_ParsingCommand_init():
    parsing_command = ParsingCommand(
        arg_indices = [0],
        command = CommandNames.IGNORE,
    )
    assert isinstance(parsing_command, ParsingCommand)

def test_Transformation_init():
    transformation = Transformation(
        arg_indices = [1],
        func = lambda x: x**2,
    )
    assert isinstance(transformation, Transformation)

def test__args_to_parser_words_no_errors(junk_tuple):
    parser_words = _args_to_parser_words(junk_tuple)
    assert len(parser_words) == 12

def test__process_array():
    pwl = _process_array(0, np.ones(7))
    assert isinstance(pwl[0], Numbers)
    assert len(pwl[0]) == 7

def test__process_data_frame(test_data_frame):
    # If the DataFrame has a non-default index, it should be treated as Labels,
    #   in an extra ParserWord following the columns.
    indexed_df = test_data_frame
    indexed_df.index = test_data_frame[test_data_frame.columns[0]]

    parser_words = _process_data_frame(0, indexed_df)

    assert isinstance(parser_words[3], Labels)

def test__process_callable():
    pwl = _process_callable(0, np.sqrt)
    assert isinstance(pwl[0], Transformation)

def test__process_dict(test_dict):
    pwl0 = _process_dict(0, test_dict)
    assert (len(pwl0) == 3)
    assert isinstance(pwl0[0], Numbers)
    assert isinstance(pwl0[1], Numbers)
    assert isinstance(pwl0[2], Labels)

    pwl1 = _process_dict(0, {'1':1, '2':2})
    assert (len(pwl1) == 1)
    assert isinstance(pwl1[0], Numbers)

    pwl2 = _process_dict(0, {'1':'one', '2':'two'})
    assert (len(pwl2) == 1)
    assert isinstance(pwl2[0], Labels)

def test__process_list():
    pwl0 = _process_list(0, [1, 3, 2, 6])
    assert (len(pwl0) == 1) and isinstance(pwl0[0], Numbers)

    pwl1 = _process_list(0, ['M', 3, None, 6])
    assert (len(pwl1) == 1) and isinstance(pwl1[0], Labels)

    pwl2 = _process_list(0, ["IGNORE", "GLOBAL"])
    assert (len(pwl2) == 2) and isinstance(pwl2[0], ParsingCommand)

def test__process_series(test_data_frame):
    pwl0 = _process_series(0, test_data_frame["y"])
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], Numbers)
    assert pwl0[0].name == "y"

    pwl0 = _process_series(0, test_data_frame["z"])
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], Labels)
    assert pwl0[0].name == "z"

def test__process_string():
    pwl0 = _process_string(0, "IGNORE")
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], ParsingCommand)

    pwl1 = _process_string(0, "RANDO")
    assert len(pwl1) == 1
    assert isinstance(pwl1[0], Labels)

def test__process_tuple(junk_tuple):
    pwl0 = _process_tuple(0, junk_tuple)
    assert len(pwl0) == 12

    pwl1 = _process_tuple(0, Color(.1, .5, .8, .3))
    assert len(pwl1) == 1
    assert isinstance(pwl1[0], Colors)
