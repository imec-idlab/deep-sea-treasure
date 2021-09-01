# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import importlib.resources as pkg_resources
from dataclasses import dataclass

from typing import Any, BinaryIO, Tuple

import os

Color = Tuple[int, int, int]


@dataclass
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

	def __eq__(self, other: Any) -> bool:
		"""
		Compare two `Theme`s and return True if they are equal.

		WARNING! In order to make sure two Themes are exactly equal, comparisons also involve reading all
		bytes from the submarine and treasure texture. If these are large texture files,
		the comparison can potentially take a long time!
		"""
		if not isinstance(other, Theme):
			return False

		def read_bytes(io: BinaryIO) -> bytes:
			"""
			Read the contents of a BinaryIO without moving the file pointer.
			"""
			current_pos = self.submarine_texture_file.tell()

			io.seek(0, os.SEEK_SET)

			byte_array: bytes = io.read()

			io.seek(current_pos, os.SEEK_SET)

			return byte_array

		return \
			(self.title == other.title) and \
			(self.sky_color == other.sky_color) and \
			(self.sea_color == other.sea_color) and \
			(self.seabed_color == other.seabed_color) and \
			(self.grid_color == other.grid_color) and \
			(self.treasure_font == other.treasure_font) and \
			(self.treasure_text_color == other.treasure_text_color) and \
			(self.debug_font == other.debug_font) and \
			(self.debug_font_size == other.debug_font_size) and \
			(self.debug_text_color == other.debug_text_color) and \
			(self.debug_text_outline_color == other.debug_text_outline_color) and \
			(read_bytes(self.submarine_texture_file) == read_bytes(other.submarine_texture_file)) and \
			(read_bytes(self.treasure_texture_file) == read_bytes(other.treasure_texture_file))

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

