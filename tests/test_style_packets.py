import os
import pytest
import numpy as np
from plotomata.style_packets import (
    StylePacket,
    SettingsPacket,
    LabelGroup,
    Color,  # indirect import resolves circular definition issue
)

# from plotomata.color_palettes import Color
from plotomata._utils import PassthroughDict, all_are_instances


@pytest.fixture
def small_label_group():
    return LabelGroup(
        keys=[0, 1],
        display_names=PassthroughDict({1: "Cyan"}),
        colors={0: Color(0, 0, 0), 1: Color(0, 255, 255)},
    )


@pytest.fixture
def medium_label_group():
    return LabelGroup(
        keys=["2", "3", "4", "5"],
        display_names=PassthroughDict({"3": "Cyan", "2": "Magenta"}),
        colors={"3": Color(0, 255, 255), "2": Color(255, 0, 255)},
    )


@pytest.fixture
def large_label_group():
    return LabelGroup(
        keys=["2", "3", "4", "5", 6, 7, 8],
        display_names=PassthroughDict({6: "Yellow", 7: "Black", 1: "yes"}),
        colors={3: Color(0, 0, 0), 7: Color(0, 0, 0), "2": Color(255, 0, 255)},
    )


@pytest.fixture
def default_settings():
    default_settings = SettingsPacket("default")
    return default_settings


@pytest.fixture
def file_out_settings():
    output_path = os.path.join(
        os.path.split(os.path.abspath(__file__))[0], "test_logs"
    )
    file_out_settings = SettingsPacket(
        "file_out",
        logging_output_path=output_path,
    )
    return file_out_settings


def test_settings_validity(default_settings, file_out_settings):
    default_settings.assert_validity()
    file_out_settings.assert_validity()


def test_logging_to_file(file_out_settings):
    file_out_settings.logger.warning("beware")


def test_StylePacket_init(small_label_group):
    sp = StylePacket(label_groups=[small_label_group])


def test_StylePacket_display_names(
    small_label_group,
    medium_label_group,
    large_label_group,
):
    sp = StylePacket(
        label_groups=[large_label_group, medium_label_group, small_label_group]
    )

    assert len(sp.display_names) == 5
    assert sp.display_names[1] == "Cyan"
