import os
import pytest
import numpy as np
from plotomata.style_packets import (
    StylePacket,
    SettingsPacket,
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
