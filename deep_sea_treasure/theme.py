# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import importlib.resources as pkg_resources
from typing import BinaryIO, Tuple

Color = Tuple[int, int, int]


class Theme:
	title: str

	sky_color: Color
	sea_color: Color
	seabed_color: Color

	grid_color: Color
	treasure_font: str
	treasure_text_color: Color
	debug_font: str
	debug_font_size: int
	debug_text_color: Color
	debug_text_outline_color: Color

	submarine_texture_file: BinaryIO
	treasure_texture_file: BinaryIO

	def __init__(
		self,
		title: str,
		sky_color: Color,
		sea_color: Color,
		seabed_color: Color,
		grid_color: Color,
		treasure_font: str,
		treasure_text_color: Color,
		debug_font: str,
		debug_font_size: int,
		debug_text_color: Color,
		debug_text_outline_color: Color,
		treasure_texture_file: BinaryIO,
		submarine_texture_file: BinaryIO,
	):
		self.title = title
		self.sky_color = sky_color
		self.sea_color = sea_color
		self.seabed_color = seabed_color
		self.grid_color = grid_color
		self.treasure_font = treasure_font
		self.treasure_text_color = treasure_text_color
		self.debug_font = debug_font
		self.debug_font_size = debug_font_size
		self.debug_text_color = debug_text_color
		self.debug_text_outline_color = debug_text_outline_color
		self.submarine_texture_file = submarine_texture_file
		self.treasure_texture_file = treasure_texture_file

	def __del__(self) -> None:
		self.submarine_texture_file.close()
		self.treasure_texture_file.close()

	@staticmethod
	def default() -> Theme:
		return Theme(
			title="DeepSeaTreasure-v0",
			sky_color=(192, 192, 255),
			sea_color=(127, 127, 255),
			seabed_color=(95, 63, 63),
			grid_color=(63, 63, 63),
			treasure_font="DejaVu Sans",
			treasure_text_color=(255, 255, 255),
			debug_font="DejaVu Sans Mono",
			debug_font_size=14,
			debug_text_color=(255, 255, 255),
			debug_text_outline_color=(0, 0, 0),
			submarine_texture_file=pkg_resources.open_binary("deep_sea_treasure.res", "submarine.png"),
			treasure_texture_file=pkg_resources.open_binary("deep_sea_treasure.res", "treasure.png")
		)

