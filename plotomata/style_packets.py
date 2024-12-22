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

    from ._utils import (
        PassthroughDict,
        possible_parser_scopes,
        all_are_instances,
    )
    from .color_palettes import Color

except ImportError as ie:
    # Alternative import style for non-standard import (source_reticulate.py).
    try:
        import sys

        sys.path.insert(0, os.path.split(os.path.abspath(__file__))[0])
        import color_palettes, _utils

        importlib.reload(color_palettes)
        importlib.reload(_utils)

        from _utils import (
            PassthroughDict,
            possible_parser_scopes,
            all_are_instances,
        )
        from color_palettes import Color

    except ImportError:
        raise ImportError(
            "plotomata failed to import component modules.\n"
        ) from ie

    except Exception as e:
        raise ImportError(
            "Unexpected error while importing plotomata component modules.\n"
        ) from e

except Exception as e:
    raise ImportError from e

import warnings
from dataclasses import dataclass
from collections.abc import Hashable
import logging
import numpy as np


@dataclass(slots=True)
class LabelGroup:
    """
    Used by StylePacket to store shared information that might be common to
    several plots for a particular document or about a particular experiment or
    project, but not plots in general (e.g. colors associated with treatment
    groups).
    """

    keys: list[Hashable]  # definitive ordered list of the members.
    display_names: PassthroughDict[Hashable, str]
    colors: dict[Hashable, Color]

    def __post_init__(self):
        # Convert display names to PassthroughDict, in case they came vanilla.
        if not isinstance(self.display_names, PassthroughDict):
            self.display_names = PassthroughDict(self.display_names)

        # Convert all colors to Color, in case the started as something else.
        for key, color in self.colors.items():
            if not isinstance(color, Color):
                self.colors[key] = Color(color)

        self.assert_validity()

    def assert_validity(self) -> None:
        assert isinstance(self.keys, list)
        assert all_are_instances(self.keys, Hashable)
        assert isinstance(self.display_names, PassthroughDict)
        if len(self.display_names) > 0:
            assert all_are_instances(self.display_names.values(), str)
            assert all_are_instances(self.display_names.keys(), Hashable)
        assert isinstance(self.colors, dict)
        if len(self.colors) > 0:
            assert all_are_instances(self.colors.values(), Color)


@dataclass(slots=True)
class StylePacket:
    """
    Stores style-related default settings that can apply to all plots. For
    example, you might want lines to always be plotted with rounded corners and
    ends and the top and right spines to always be hidden. One StylePacket
    defining those choices can be passes to all plotters to ensure consistent
    formatting.
    """

    label_groups: list[LabelGroup]  # last overrides previous
    outer_margin: tuple[float, float] = (0.1, 0.1)
    inner_margin: tuple[float, float] = (0.1, 0.1)
    subplot_margins: tuple[float, float] = (0.2, 0.2)

    def __post_init__(self):
        self.label_groups = list(self.label_groups)
        self.outer_margin = (self.outer_margin[0], self.outer_margin[1])
        self.assert_validity()

    def assert_validity(self):
        assert isinstance(self.label_groups, list)
        assert all_are_instances(self.label_groups, LabelGroup)
        assert isinstance(self.outer_margin, tuple)
        assert len(self.outer_margin) == 2
        assert all_are_instances(self.outer_margin, float)
        assert self.outer_margin[0] >= 0 and self.outer_margin[1] >= 0
        assert isinstance(self.inner_margin, tuple)
        assert len(self.inner_margin) == 2
        assert all_are_instances(self.inner_margin, float)
        assert self.inner_margin[0] >= 0 and self.inner_margin[1] >= 0

    @property
    def display_names(self) -> PassthroughDict:
        # This will behave as if StylePacket had a single display names
        # PassthroughDict, but it actually makes a combined PassthroughDict from
        # all LabelGroups.
        combined_disp_names = PassthroughDict({})
        for lg in self.label_groups:
            for key, value in lg.display_names.items():
                combined_disp_names[key] = value

        return combined_disp_names

    def list_display_names(
        self,
        keys: list[Hashable],
    ) -> list[Hashable]:
        """
        Given a list of possible keys, this function will convert them to
        display names using only the only label group that contaings the most
        keys.
        """
        assert isinstance(keys, list)
        assert all_are_instances(keys, Hashable)

        best_dict = PassthroughDict({})
        best_score = 0
        for lg in self.label_groups:
            # select the label group with the most matching keys
            score = np.sum([key in lg.display_names for key in keys])
            if score >= best_score:
                best_score = score
                best_dict = lg.display_names

        return [best_dict[key] for key in keys]

    def list_colors(
        self,
        keys: list[Hashable],
    ) -> list[Hashable]:
        """
        Given a list of possible keys, this function will convert them to
        display names using only the only label group that contaings the most
        keys.
        """
        assert isinstance(keys, list)
        assert all_are_instances(keys, Hashable)

        best_colors = {}
        best_score = 0
        for lg in self.label_groups:
            # select the label group with the most matching keys
            score = np.sum([key in lg.colors for key in keys])
            if score >= best_score:
                best_score = score
                best_colors = lg.colors

        return [best_colors[key] for key in keys]


logging_level_from_string = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


@dataclass
class SettingsPacket:
    """
    Stores settings to determine various behaviors of plotomata and is
    responsible for logging.
    """

    name: str = "<unnamed>"
    default_parser_scope: str = "next_hit"
    expert_mode: bool = False  # for risker but more powerful behavior
    logging_output_path: os.PathLike | None = None
    _logging_level: int = logging.INFO
    skip_asserts: bool = False

    @property
    def logging_level(self):
        return self._logging_level

    @logging_level.setter
    def logging_level(self, level):
        self._logging_level = level
        self.logger.setLevel(self.logging_level)

    def __post_init__(self):
        # logger setup
        self.logger = logging.getLogger(self.name)
        if isinstance(self.logging_output_path, (os.PathLike, str)):
            if not os.path.exists(self.logging_output_path):
                if self.expert_mode:
                    os.makedirs(self.logging_output_path)
                else:
                    raise FileNotFoundError(
                        f"logging_output_path {self.logging_output_path} does"
                        + "not exist. Either set a location that exists or set"
                        + "expert_mode=True to automatically create it."
                    )
            self.log_handler = logging.FileHandler(
                os.path.join(self.logging_output_path, "plotomata.log")
            )
        elif self.logging_output_path is None:
            self.log_handler = logging.StreamHandler()
        else:
            raise TypeError(
                "logging_output_path should be os.PathLike or str, not "
                + f"{self.logging_output_path} "
                + f"(of type: {type(self.logging_output_path)})."
            )
        self.log_handler.setLevel(self.logging_level)
        self.log_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_handler.setFormatter(self.log_formatter)
        self.logger.addHandler(self.log_handler)

        if self.logging_level in logging_level_from_string:
            self.logging_level = logging_level_from_string[self.logging_level]

        self.assert_validity()

        return self

    def assert_validity(self):
        assert self.default_parser_scope in possible_parser_scopes
        assert isinstance(self.expert_mode, bool)
        assert (self.logging_output_path is None) or isinstance(
            self.logging_output_path, (os.PathLike, str)
        )
        assert isinstance(self.logger, logging.Logger)
        assert isinstance(self.log_handler, logging.Handler)
        assert isinstance(self.log_formatter, logging.Formatter)
        assert isinstance(self.name, str)
        assert self.logging_level in logging_level_from_string.values()
