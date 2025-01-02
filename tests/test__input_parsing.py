import warnings
import pytest
import pandas as pd
import numpy as np
from plotomata._input_parsing import (
    Numeric,
    Labels,
    Colors,
    Command,
    Transformation,
    CommandNames,
    ParserState,
    PlotDataRequirements,
    Matrix,
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
from plotomata._utils import PassthroughDict, ScopeModes
from plotomata.color_palettes import Color
from plotomata.style_packets import SettingsPacket


@pytest.fixture
def numbers_pw():
    return Numeric(
        arg_indices=[0],
        numbers=np.array([1, 2, 3]),
        name="counting",
    )


@pytest.fixture
def test_dict():
    return {"x": [1, 2, 3], "y": [4, 2, 3], "z": ["five", "seven", "six"]}


@pytest.fixture
def test_data_frame(test_dict):
    return pd.DataFrame(test_dict)


@pytest.fixture
def junk_tuple(test_dict, test_data_frame):
    return (
        test_data_frame,  # produces 3 p.w.
        test_dict["x"],
        PassthroughDict({"x": 7, "y": 7, "z": 7}),
        np.array([6, 6, 6, 7]),
        ["a", "b", "c", "d"],
        Color((1, 0, 1)),
        "READ_AS_COLOR",
        (
            np.array([6, 6, 6, 7]),
            ["a", "b", "c", "d"],
            np.array([6, 16, 6, 7]),
        ),  # produces 2 ParserWords
        lambda a: np.sqrt(a),
        "NEXT_ARG",
        "IGNORE",
        test_dict["z"],
    )


@pytest.fixture
def monotone_scatter_directive():
    return PlotDataRequirements.monochrome_scatter()


@pytest.fixture
def default_parser_sate():
    settings = SettingsPacket()
    return ParserState.initial_state(settings)


def test_Numeric_to_Colors_conversion(numbers_pw):
    colors_pw = numbers_pw.to_color_if_possible()
    assert isinstance(colors_pw, Colors)


def test_Labels_to_Colors_conversion():
    hex_color_labels_pw = _process_string(1, "#FF11CC")[0]
    colors_pw = hex_color_labels_pw.to_color_if_possible()
    assert isinstance(colors_pw, Colors)


def test_Labels_init():
    labels = Labels(
        arg_indices=[0],
        labels=[1, None, "three"],
        name="counting",
    )
    assert isinstance(labels, Labels)


def test_Colors_init():
    colors = Colors(
        arg_indices=[0],
        colors=[Color(1, 2, 3)],
        name="dark",
    )
    assert isinstance(colors, Colors)


def test_Command_init():
    parsing_command = Command(
        arg_indices=[0],
        command=CommandNames.IGNORE,
    )
    assert isinstance(parsing_command, Command)


def test_Transformation_init():
    transformation = Transformation(
        arg_indices=[1],
        func=lambda x: x**2,
    )
    assert isinstance(transformation, Transformation)


def test__args_to_parser_words_no_errors(default_parser_sate, junk_tuple):
    settings = SettingsPacket()
    parser_words = _args_to_parser_words(
        default_parser_sate, settings, junk_tuple
    )
    assert len(parser_words) == 14


def test__process_array(default_parser_sate):
    pwl = _process_array(0, np.ones(7), default_parser_sate)
    assert isinstance(pwl[0], Numeric)
    assert len(pwl[0]) == 7


def test__process_data_frame(test_data_frame, default_parser_sate):
    # If the DataFrame has a non-default index, it should be treated as Labels,
    #   in an extra ParserWord following the columns.
    indexed_df = test_data_frame
    indexed_df.index = test_data_frame[test_data_frame.columns[0]]

    parser_words = _process_data_frame(0, indexed_df, default_parser_sate)

    assert isinstance(parser_words[3], Labels)

    matrix_parser_state = default_parser_sate
    matrix_parser_state.read_as_matrix = True

    matrix_df = test_data_frame[["x", "y"]]
    matrix_df.index = test_data_frame["z"]

    parser_words = _process_data_frame(0, matrix_df, matrix_parser_state)

    assert isinstance(parser_words[0], Matrix)


def test__process_callable():
    pwl = _process_callable(0, np.sqrt)
    assert isinstance(pwl[0], Transformation)


def test__process_dict(test_dict, default_parser_sate):
    settings = SettingsPacket()

    pwl0 = _process_dict(0, test_dict, default_parser_sate, settings)
    assert len(pwl0) == 3
    assert isinstance(pwl0[0], Numeric)
    assert isinstance(pwl0[1], Numeric)
    assert isinstance(pwl0[2], Labels)

    pwl1 = _process_dict(0, {"1": 1, "2": 2}, default_parser_sate, settings)
    assert len(pwl1) == 1
    assert isinstance(pwl1[0], Numeric)

    pwl2 = _process_dict(
        0, {"1": "one", "2": "two"}, default_parser_sate, settings
    )
    assert len(pwl2) == 1
    assert isinstance(pwl2[0], Labels)


def test__process_list(default_parser_sate):
    settings = SettingsPacket()

    pwl0 = _process_list(0, [1, 3, 2, 6], default_parser_sate, settings)
    assert (len(pwl0) == 1) and isinstance(pwl0[0], Numeric)

    pwl1 = _process_list(0, ["M", 3, None, 6], default_parser_sate, settings)
    assert (len(pwl1) == 1) and isinstance(pwl1[0], Labels)

    pwl2 = _process_list(0, ["IGNORE", "GLOBAL"], default_parser_sate, settings)
    assert (len(pwl2) == 0)


def test__process_series(test_data_frame):
    pwl0 = _process_series(0, test_data_frame["y"])
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], Numeric)
    assert pwl0[0].name == "y"

    pwl0 = _process_series(0, test_data_frame["z"])
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], Labels)
    assert pwl0[0].name == "z"


def test__process_string():
    pwl0 = _process_string(0, "IGNORE")
    assert len(pwl0) == 1
    assert isinstance(pwl0[0], Command)

    pwl1 = _process_string(0, "RANDO")
    assert len(pwl1) == 1
    assert isinstance(pwl1[0], Labels)


def test__process_tuple(junk_tuple, default_parser_sate):
    settings = SettingsPacket()
    pwl0 = _process_tuple(0, junk_tuple, default_parser_sate, settings)
    assert len(pwl0) == 14

    pwl1 = _process_tuple(
        0,
        Color(0.1, 0.5, 0.8, 0.3),
        default_parser_sate,
        settings,
    )
    assert len(pwl1) == 1
    assert isinstance(pwl1[0], Colors)


def test_ParserState_init():
    ps1 = ParserState(default_scope=ScopeModes.GLOBAL)
    settings = SettingsPacket(parser_default_scope=ScopeModes.GLOBAL)
    ps2 = ParserState.initial_state(settings)
    assert ps1 == ps2


def test_ParserState_obey():
    settings = SettingsPacket(parser_default_scope=ScopeModes.NEXT_ARG)
    ps = ParserState.initial_state(settings)
    assert ps.scope == ScopeModes.NEXT_ARG
    ps.obey(Command([0], CommandNames.GLOBAL))
    assert ps.scope == ScopeModes.GLOBAL


def test_ParserState_possible_referents_mask(default_parser_sate, junk_tuple):
    settings = SettingsPacket()
    words = _args_to_parser_words(default_parser_sate, settings, *junk_tuple)
    settings = SettingsPacket(parser_default_scope=ScopeModes.NEXT_HIT)
    ps = ParserState.initial_state(settings)
    mask = ps.possible_referents_mask(words, 8)
    assert np.sum(mask) == 6

    settings = SettingsPacket(parser_default_scope=ScopeModes.NEXT_ARG)
    ps = ParserState.initial_state(settings)
    mask = ps.possible_referents_mask(words, 8)
    assert np.sum(mask) == 2

    settings = SettingsPacket(parser_default_scope=ScopeModes.GLOBAL)
    ps = ParserState.initial_state(settings)
    mask = ps.possible_referents_mask(words, 8)
    assert np.sum(mask) == 6


def test_PlotDataRequirement_check_fulfillment(
    monotone_scatter_directive, test_data_frame, default_parser_sate
):
    settings = SettingsPacket()
    x, y, z = tuple(
        _args_to_parser_words(
            default_parser_sate,
            settings,
            test_data_frame,
        )
    )
    assert monotone_scatter_directive.check_fulfillment(
        data={"x": x, "y": y}
    )

    # no unexpected keys
    assert not monotone_scatter_directive.check_fulfillment(
        data={"x": x, "y": y, "z": z}
    )

    assert monotone_scatter_directive.check_fulfillment(
        data={"x": x, "y": y, "point_names": z},
    )

    # strings can't be size_data
    assert not monotone_scatter_directive.check_fulfillment(
        data={"x": x, "y": y, "size_data": z},
    )

    # strings can't be position data
    assert not monotone_scatter_directive.check_fulfillment(
        data={"x": x, "y": z},
    )


def test_state_scope_management_first_pass():
    pass