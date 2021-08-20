# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pygame

from typing import Dict, List, Tuple


class OutlinedFont:
	normal_font: pygame.font.Font
	circle_cache: Dict[int, List[Tuple[int, int]]]

	def __init__(self, font_name: str, font_size: int, italic: bool):
		self.normal_font = pygame.font.SysFont(font_name, font_size, bold=False, italic=italic)
		self.circle_cache = {}

	def __circle_points(self, radius: int) -> List[Tuple[int, int]]:
		if radius in self.circle_cache:
			return self.circle_cache[radius]

		x: int = radius
		y: int = 0
		e: int = 1 - radius

		points: List[Tuple[int, int]] = []
		self.circle_cache[radius] = points

		while x >= y:
			points.append((x, y))
			y += 1
			if e < 0:
				e += (2 * y) - 1
			else:
				x -= 1
				e += 2 * (y - x) - 1
		points += [(y, x) for x, y in points if x > y]
		points += [(-x, y) for x, y in points if x]
		points += [(x, -y) for x, y in points if y]
		points.sort()
		return points

	def render(self, text: str, antialias: bool, color: Tuple[int, int, int], outline_color: Tuple[int, int, int], outline_radius: int) -> pygame.surface.Surface:
		text_surface: pygame.surface.Surface = self.normal_font.render(text, antialias, color).convert_alpha()

		width: int = text_surface.get_width() + (2 * outline_radius)
		height: int = text_surface.get_height()

		outline = pygame.Surface((width, height + (2 * outline_radius))).convert_alpha()
		outline.fill((0, 0, 0, 0))

		output_surface = outline.copy()

		outline.blit(self.normal_font.render(text, True, outline_color).convert_alpha(), (0, 0))

		for (dx, dy) in self.__circle_points(outline_radius):
			output_surface.blit(outline, (dx + outline_radius, dy + outline_radius))

		output_surface.blit(text_surface, (outline_radius, outline_radius))

		return output_surface
