# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import numpy.typing as npt
import pygame

from typing import Any, BinaryIO, Dict, Optional, Tuple, List

from deep_sea_treasure.theme import Theme
from deep_sea_treasure.outlined_font import OutlinedFont
from deep_sea_treasure.contract import contract


class DeepSeaTreasureV0Renderer:
	# Color/Style theme
	theme: Theme

	# Dimensions
	x_tiles: int
	y_tiles: int

	tile_width: int
	tile_height: int

	# PyGame objects
	display: pygame.surface.Surface

	submarine_texture: pygame.surface.Surface
	treasure_texture: pygame.surface.Surface

	debug_font: OutlinedFont
	treasure_font: pygame.font.Font

	def __init__(self, theme: Theme, tile_width: int, tile_height: int, x_tiles: int, y_tiles: int):
		contract(pygame.image.get_extended(), "Your Pygame installation might not support the PNG file format! See https://www.pygame.org/docs/ref/image.html#pygame.image.get_extended for more information.")
		contract(0 < x_tiles, "x_tiles must be > 0, got {0}!", x_tiles)
		contract(0 < y_tiles, "y_tiles must be > 0, got {0}!", y_tiles)
		contract(0 < tile_width, "tile_width must be > 0, got {0}!", tile_width)
		contract(0 < tile_height, "tile_height must be > 0, got {0}!", tile_height)

		self.x_tiles = x_tiles
		self.y_tiles = y_tiles

		self.tile_width = tile_width
		self.tile_height = tile_height

		self.theme = theme

		success, failed = pygame.init()

		contract(0 == failed, "Failed to initialize {0} pygame modules ({1})! ({2} modules were initialized successfully)", failed, ', '.join(['pygame.' + mod for mod in self.__list_failed_modules()]), success)

		self.display = pygame.display.set_mode(size=(tile_width * x_tiles, tile_height * (y_tiles + 2)))
		pygame.display.set_caption(self.theme.title)

		self.submarine_texture = pygame.transform.scale(self.__load_texture(self.theme.submarine_texture_file), (self.tile_width, self.tile_height))
		self.treasure_texture = pygame.transform.scale(self.__load_texture(self.theme.treasure_texture_file), (self.tile_width, self.tile_height))
		self.treasure_font = self.__load_font(self.theme.treasure_font, bold=True)
		self.debug_font = OutlinedFont(theme.debug_font, self.theme.debug_font_size, italic=False)

	def __load_texture(self, file: BinaryIO) -> pygame.surface.Surface:
		return pygame.image.load(file).convert_alpha()

	def __load_font(self, font_name: str, bold: bool = False) -> pygame.font.Font:
		return pygame.font.SysFont(font_name, 16, bold)

	def __list_failed_modules(self) -> List[str]:
		all_modules = {
			"display": pygame.display,
			"font": pygame.font,
			"joystick": pygame.joystick,
		}

		failed_modules: List[str] = []

		for module in all_modules:
			if not all_modules[module].get_init(): #type: ignore[attr-defined]
				failed_modules.append(module)

		return failed_modules

	def __render_debug_text(self, debug_info: Dict[str, Any]) -> List[pygame.surface.Surface]:
		outline_radius: int = 2

		strings: List[str] = [
			"position:",
			f" x: {debug_info['position']['x']:3d}",
			f" y: {debug_info['position']['y']:3d}",
			"velocity:",
			f" x: {debug_info['velocity']['x']:3d}",
			f" y: {debug_info['velocity']['y']:3d}",
			"collision:",
			f" horizontal: {1 if debug_info['collision']['horizontal'] else 0:1d}",
			f" vertical: {1 if debug_info['collision']['vertical'] else 0:1d}",
			f" diagonal: {1 if debug_info['collision']['diagonal'] else 0:1d}"
		]

		return [
			self.debug_font.render(txt, True, self.theme.debug_text_color, self.theme.debug_text_outline_color, outline_radius) for txt in strings
		]

	def render(self,
		submarines: List[Tuple[float, float]],
		seabed: npt.NDArray[np.single],
		treasure_values: Dict[Tuple[int, int], float],
		debug_info: Optional[Dict[str, Any]] = None,
		render_grid: bool = False,
		render_treasure_values: bool = False
	) -> None:
		contract(seabed.shape[-1] == self.x_tiles,
			"Renderer was constructed for {0}x{1} environment, but seabed implied {2}x{3} environment", self.x_tiles, self.y_tiles, seabed.shape[-1], int(np.max(seabed[0]) - np.min(seabed[0])))	# type: ignore[no-untyped-call]
		contract(np.max(seabed[0]) <= self.y_tiles,																																					# type: ignore[no-untyped-call]
			"Renderer was constructed for {0}x{1} environment, but seabed implied {2}x{3} environment", self.x_tiles, self.y_tiles, seabed.shape[-1], int(np.max(seabed[0]) - np.min(seabed[0])))	# type: ignore[no-untyped-call]

		# Draw the sky
		for x in range(seabed.shape[-1]):
			pygame.draw.rect(self.display, self.theme.sky_color, (x * self.tile_width, 0, self.tile_width, self.tile_height))

		# Draw all sea tiles
		for x in range(self.x_tiles):
			for y in range(0, int(seabed[x]) + 1):
				pygame.draw.rect(self.display, self.theme.sea_color, (x * self.tile_width, (y + 1) * self.tile_height, self.tile_width, self.tile_height))

		# Draw the seabed
		for x in range(self.x_tiles):
			for y in range(int(seabed[x]) + 1, self.y_tiles + 1):
				pygame.draw.rect(self.display, self.theme.seabed_color, (x * self.tile_width, (y + 1) * self.tile_height, self.tile_width, self.tile_height))

		# Render treasures
		for treasure_x in range(seabed.shape[-1]):
			treasure_y: int = int(seabed[treasure_x])

			if (treasure_x, treasure_y) in treasure_values:
				self.display.blit(self.treasure_texture, (treasure_x * self.tile_width, (treasure_y + 1) * self.tile_height, self.tile_width, self.tile_height))

				if render_treasure_values:
					treasure_text = self.treasure_font.render(f"{treasure_values[(treasure_x, treasure_y)]:.2f}", True, self.theme.treasure_text_color)

					text_width: int = treasure_text.get_width()
					text_height: int = treasure_text.get_height()

					text_x_offset: int = (self.tile_width - text_width) // 2
					text_y_offset: int = self.tile_height - text_height

					self.display.blit(treasure_text, ((treasure_x * self.tile_width) + text_x_offset, ((treasure_y + 1) * self.tile_height) + text_y_offset, text_width, text_height))

		# Draw all submarines
		for sub in submarines:
			sub_x, sub_y = sub

			self.display.blit(self.submarine_texture, (sub_x * self.tile_width, (sub_y + 1) * self.tile_height, self.tile_width, self.tile_height))

		# If the user asked for it, render the grid
		if render_grid:
			for tile_x in range(seabed.shape[-1]):
				pygame.draw.line(self.display, self.theme.grid_color, ((tile_x + 1) * self.tile_width, 0), ((tile_x + 1) * self.tile_width, self.display.get_height()))

			for tile_y in range(1, int(self.display.get_height() / self.tile_height)):
				pygame.draw.line(self.display, self.theme.grid_color, (0, (tile_y * self.tile_height)), (self.display.get_width(), (tile_y * self.tile_height)))

		# If the user supplied debugging information, render this as well.
		if debug_info is not None:
			debug_dict: Dict[str, Any] = debug_info

			# We first throw away all debugging data from wrappers, and keep only the dict from the environment itself.
			while debug_dict["env"] != "DeepSeaTreasureV0":
				debug_dict = debug_dict["inner"]

			debug_lines: List[pygame.surface.Surface] = self.__render_debug_text(debug_dict)

			max_width: int = max([line.get_width() for line in debug_lines])

			text_y: int = 0
			text_x: int = self.display.get_width() - max_width

			for line in debug_lines:
				self.display.blit(line, (text_x, text_y, line.get_width(), line.get_height()))

				text_y += line.get_height()

			if (debug_dict["velocity"]["x"] != 0) or (debug_dict["velocity"]["y"] != 0):
				x = (debug_dict["position"]["x"] * self.tile_width) + (self.tile_width / 2.0)
				y = ((debug_dict["position"]["y"] + 1) * self.tile_height) + (self.tile_height / 2.0)

				dx = debug_dict["velocity"]["x"] * self.tile_width
				dy = debug_dict["velocity"]["y"] * self.tile_height

				pygame.draw.line(self.display, (255, 0, 0), (x, y), (x + dx, y + dy), 2)

		pygame.display.flip()
