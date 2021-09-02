# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import gym
import json
import numpy as np
import numpy.typing as npt
import importlib.resources as pkg_resources

from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

from jsonschema import Draft7Validator

from deep_sea_treasure.deep_sea_treasure_v0_renderer import DeepSeaTreasureV0Renderer
from deep_sea_treasure.theme import Theme
from deep_sea_treasure.contract import contract


class DeepSeaTreasureV0(gym.Env): #type: ignore[misc]
	"""
	gym-compatible environment designed for research into Multi-Objective Reinforcement Learning.

	The recommend way to create an instance of this environment is through the static `new()` method, rather than by directly calling the constructor.
	The constructor takes a dictionary for configuration, to preserve compatibility with frameworks such as RLLib/Ray.
	"""
	# A space representing all possible rewards, similar to an observation space or an action space
	reward_space: gym.Space

	acceleration_levels: npt.NDArray[np.int32]

	# Environment
	seabed: npt.NDArray[np.single]
	implicit_collision_objective: bool
	treasures: Dict[Tuple[int, int], float]

	# Submarine (position and velocity)
	max_vel: npt.NDArray[np.int32]
	min_vel: npt.NDArray[np.int32]

	sub_pos: npt.NDArray[np.int32]
	sub_vel: npt.NDArray[np.int32]

	# Time
	time_step: int
	max_time_steps: int

	# Rendering
	render_grid: bool
	render_treasure_values: bool
	theme: Theme
	renderer: Optional[DeepSeaTreasureV0Renderer]

	def __init__(self, env_config: Dict[str, Any]):
		super(DeepSeaTreasureV0, self).__init__()

		Draft7Validator(schema=DeepSeaTreasureV0.schema()).validate(env_config)

		contract(float(env_config["max_velocity"]) < float(len(env_config["treasure_values"])), "Maximum velocity ({0}) can never exceed size of world ({1})!", env_config['max_velocity'], len(env_config['treasure_values']))

		# Action space: 2 x no. acceleration levels.
		# One is acceleration in x-direction, other is acceleration in y-direction.
		# there are a number of actions in each direction, each corresponding to a single level of acceleration.
		self.action_space = gym.spaces.Tuple(
			(gym.spaces.Discrete((2 * len(env_config["acceleration_levels"])) + 1), gym.spaces.Discrete((2 * len(env_config["acceleration_levels"])) + 1))
		)

		config_accels: List[int] = sorted([int(l) for l in env_config["acceleration_levels"]])

		# Store the mapping from discrete actions to acceleration values
		self.acceleration_levels = np.concatenate([						# type: ignore[no-untyped-call]
			-np.asarray(list(reversed(config_accels)), dtype=np.int32),
			np.zeros((1,), dtype=np.int32),
			np.asarray(config_accels, dtype=np.int32)
		], dtype=np.int32)

		contract(self.acceleration_levels.dtype == np.int32, "Acceleration levels should have datatype {0}, got {1}", np.int32, self.acceleration_levels.dtype)

		# Whether or not a collision should cause a drastic drop in reward values
		self.implicit_collision_objective = bool(env_config["implicit_collision_constraint"])

		# Observation is a 2 x (N + 1) matrix, if the environment has size N
		# First column is the submarine's x (0) and y (1) velocity
		# Next N columns represent the relative coordinates from the submarine to each treasure
		self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("+inf"), shape=(2, 1 + len(env_config["treasure_values"])))

		# Dictionary
		# This dictionary maps an (x, y) coordinate pair to the associated treasure
		# If the coordinate pair does not exist in the dictionary, then this square does not contain treasure
		self.treasures = {}

		x_set: Set[int] = set()

		seabed_coordinates: List[Tuple[int, int]] = []

		for t in env_config["treasure_values"]:
			xy_list = t[0]
			treasure: float = float(t[1])

			x: int = int(xy_list[0])
			y: int = int(xy_list[1])

			contract(x not in x_set, "Every x-value can occur only once in treasure list! x-value {0} occured more than once in treasure list!", x)

			self.treasures[(x, y)] = treasure
			x_set.add(x)

			seabed_coordinates.append((x, y))

		seabed_coordinates = sorted(seabed_coordinates)

		max_x: int = max([x for (x, y) in seabed_coordinates])

		# 1 x N array
		# This array contains the height of the seabed at every x-coordinate
		self.seabed = np.ndarray(shape=(max_x + 1,), dtype=np.single)

		seabed_index: int = 0

		for x in range(max_x + 1):
			if (seabed_coordinates[seabed_index][0] < x) and ((seabed_index + 1) < len(seabed_coordinates)):
				seabed_index = seabed_index + 1

			self.seabed[x] = seabed_coordinates[seabed_index][1]

		# Reward is a 2 x N matrix
		# The maximum value is the highest possible reward, minimum value is determined by time-reward, which is infinite
		reward_low: npt.NDArray[np.single] = np.asarray([0.0, -1.0])

		if env_config["implicit_collision_constraint"]:
			reward_low -= 1.0

		self.reward_space = gym.spaces.Box(low=reward_low, high=np.asarray([max(self.treasures.values()), -1.0]), shape=(2,), dtype=np.float32)

		# Minimum/Maximum velocity:
		self.max_vel = np.asarray([[float(env_config["max_velocity"])], [float(env_config["max_velocity"])]], dtype=np.int32)
		self.min_vel = -np.asarray([[float(env_config["max_velocity"])], [float(env_config["max_velocity"])]], dtype=np.int32)

		# Coordinates of the submarine are given in (x, y) form, top-left of the environment is (0, 0)
		self.sub_pos = np.zeros((2, 1), dtype=np.int32)
		self.sub_vel = np.zeros((2, 1), dtype=np.int32)

		# How many timesteps have passed since the start of the episode
		self.time_step = 0

		# How many timesteps an agent is allowed to take before the episode ends
		self.max_time_steps = int(env_config["max_steps"])

		# Rendering options and renderer
		self.render_grid = bool(env_config["render_grid"])
		self.render_treasure_values = bool(env_config["render_treasure_values"])

		self.theme = env_config["theme"]
		self.renderer = None

		self.reset()

	def __debug(self) -> Dict[str, Any]:
		return {
			"env": self.__class__.__name__,
			"treasures": int(len(self.treasures)),
			"position": {
				"x": int(self.sub_pos[0]),
				"y": int(self.sub_pos[1])
			},
			"time": {
				"current": self.time_step,
				"max": self.max_time_steps
			},
			"velocity": {
				"x": int(self.sub_vel[0]),
				"y": int(self.sub_vel[1]),
				"max_x": int(self.max_vel[0]),
				"max_y": int(self.max_vel[1]),
				"min_x": int(self.min_vel[0]),
				"min_y": int(self.min_vel[1])
			},
			"collision": {
				"horizontal": False,
				"vertical": False
			}
		}

	def __observe(self) -> npt.NDArray[np.int32]:
		treasure_coords: npt.NDArray[np.int32] = np.asarray(sorted(list(self.treasures.keys()))).transpose()

		relative_treasure_coords: npt.NDArray[np.int32] = treasure_coords - np.tile(self.sub_pos, (1, treasure_coords.shape[1]))	# type: ignore[no-untyped-call]

		return cast(npt.NDArray[np.int32], np.concatenate([self.sub_vel, relative_treasure_coords], axis=1, dtype=np.int32))	# type: ignore[no-untyped-call]

	def __get_rewards(self, collides: bool) -> npt.NDArray[np.single]:
		rewards: npt.NDArray[np.single] = np.zeros((2,), dtype=np.single)

		# Treasure reward
		if self.__is_done():
			rel_pos: npt.NDArray[np.single] = self.__observe()[:, 1:]
			found_treasure: npt.NDArray[np.bool_] = cast(npt.NDArray[np.bool_], np.all(rel_pos == 0, axis=0))

			# Check that the environment ended because we found treasure, and not for some other reason.
			if 0 < int(np.sum(found_treasure)):
				treasure_index = int(np.argmax(found_treasure))
				treasure_coords: Tuple[int, int] = sorted(self.treasures.keys())[treasure_index]

				rewards[0] = self.treasures[treasure_coords]
		# Time reward
		rewards[1] = -1

		if self.implicit_collision_objective and collides:
			rewards = self.reward_space.low

		return rewards

	def __is_done(self) -> bool:
		if self.max_time_steps <= self.time_step:
			return True

		# Check to see if we're on a treasure
		# First, extract our position relative to all treasures from the current observation
		rel_positions: npt.NDArray[np.single] = self.__observe()[:, 1:]

		# See: https://stackoverflow.com/a/14860884
		# Compare each value to the element in the first row
		tmp = rel_positions == 0.0

		# Reduce over vertical axis using an AND-function (i.e., all elements in the column should be True for this to be True)
		red = np.all(tmp, axis=0)

		# If after reduction, any element is True, return True, otherwise return False
		return bool(np.any(red))

	def __get_left_wall(self, x: int, y: int) -> int:
		for x in range(int(x), -1, -1):
			if self.seabed[x] < float(y):
				return x

		return 0

	def __get_right_wall(self, x: int, y: int) -> int:
		for x in range(int(x), self.seabed.shape[0]):
			if self.seabed[x] < float(y):
				return x

		return self.seabed.shape[0]

	def __get_bottom_wall(self, x: int) -> int:
		return int(self.seabed[x])

	def __collides_horizontal(self, next_pos: npt.NDArray[np.int32], left_wall: int, right_wall: int) -> bool:
		if (next_pos[0] < 0) or ((self.seabed.shape[0] - 1) < next_pos[0]):
			return True

		return bool(next_pos[0] < left_wall) or bool(right_wall < next_pos[0])

	def __collides_vertical(self, next_pos: npt.NDArray[np.int32], bottom_wall: int) -> bool:
		if (next_pos[1] < 0) or (int(np.max(self.seabed)) < next_pos[1]):	# type: ignore[no-untyped-call]
			return True

		return bool(bottom_wall < next_pos[1])

	def __collides_diagonally(self, next_pos: npt.NDArray[np.int32]) -> bool:
		seabed_y: int = int(self.seabed[next_pos[0]])

		return bool(seabed_y < next_pos[1])

	def step(self, action: Union[Tuple[int, int], Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.single], bool, Dict[str, Any]]:
		x_action, y_action = action

		if isinstance(x_action, int):
			contract((0 <= x_action) and (x_action < self.action_space[0].n), "Integer action must be in range [{0}, {1}[", 0, self.action_space[0].n)

			index = x_action
			x_action = np.zeros(shape=(self.action_space[0].n,), dtype=np.int32)
			x_action[index] = 1

		if isinstance(y_action, int):
			contract((0 <= y_action) and (y_action < self.action_space[1].n), "Integer action must be in range [{0}, {1}[", 0, self.action_space[1].n)

			index = y_action
			y_action = np.zeros(shape=(self.action_space[1].n,), dtype=np.int32)
			y_action[index] = 1

		contract(isinstance(x_action, np.ndarray), "Action must be {0} after action normalization!", np.ndarray.__class__.__name__)
		contract(isinstance(y_action, np.ndarray), "Action must be {0} after action normalization!", np.ndarray.__class__.__name__)

		contract(not bool(np.any(np.isnan(x_action))), "X-Action contained NaNs: {0}!", x_action)
		contract(not bool(np.any(np.isnan(y_action))), "Y-Action contained NaNs: {0}!", y_action)

		contract(len(x_action) == len(self.acceleration_levels), "Got X-action with {0} possible actions, expected {1} possible actions!", len(x_action), len(self.acceleration_levels))
		contract(len(y_action) == len(self.acceleration_levels), "Got Y-action with {0} possible actions, expected {1} possible actions!", len(y_action), len(self.acceleration_levels))

		contract(1 == int(np.sum(x_action)), "X-Action must be 1-hot encoded, got {0}!", x_action)
		contract(1 == int(np.sum(y_action)), "Y-Action must be 1-hot encoded, got {1}!", y_action)

		x_accel = self.acceleration_levels[int(np.argmax(x_action))]
		y_accel = self.acceleration_levels[int(np.argmax(y_action))]

		self.sub_vel += np.expand_dims(np.asarray([x_accel, y_accel]), 1)	# type: ignore[no-untyped-call]

		# Clip velocity to make sure the agent can't perform any physics shenanigans
		self.sub_vel = cast(npt.NDArray[np.int32], np.clip(self.sub_vel, self.min_vel, self.max_vel))

		next_pos = self.sub_pos + self.sub_vel

		# Check if moving in a straight line would already cause a collision
		left_wall: int = self.__get_left_wall(self.sub_pos[0], self.sub_pos[1])
		right_wall: int = self.__get_right_wall(self.sub_pos[0], self.sub_pos[1])
		bottom_wall: int = self.__get_bottom_wall(self.sub_pos[0])

		horizontal_collision = self.__collides_horizontal(next_pos=next_pos, left_wall=left_wall, right_wall=right_wall)
		vertical_collision = self.__collides_vertical(next_pos=next_pos, bottom_wall=bottom_wall)

		diagonal_collision: bool = False

		# Check if the resulting square is inaccessible
		if (not horizontal_collision) and (not vertical_collision):
			diagonal_collision = self.__collides_diagonally(next_pos=next_pos)

		# Assert an invariant that diagonal collision can never be true is either of the other collisions are true
		contract(False == ((horizontal_collision or vertical_collision) and diagonal_collision), "Diagonal collision ({0:1d}) can only be true if neither horizontal ({1:1d}) nor vertical ({2:1d}) collisions are true", int(diagonal_collision), int(horizontal_collision), int(vertical_collision))

		collision: bool = horizontal_collision or vertical_collision or diagonal_collision

		# If we clip in either direction, zero the velocity
		if collision:
			self.sub_vel = np.zeros_like(self.sub_vel)

		self.sub_pos += self.sub_vel

		# Increment time
		self.time_step += 1

		# Indicate if this action caused a collision
		debug_dict: Dict[str, Any] = self.__debug()
		debug_dict["collision"]["horizontal"] = horizontal_collision
		debug_dict["collision"]["vertical"] = vertical_collision
		debug_dict["collision"]["diagonal"] = diagonal_collision

		return self.__observe(), self.__get_rewards(collides=collision), self.__is_done(), debug_dict

	def reset(self) -> npt.NDArray[np.int32]:
		self.sub_pos = np.zeros_like(self.sub_pos, dtype=np.int32)
		self.sub_vel = np.zeros_like(self.sub_vel, dtype=np.int32)
		self.time_step = 0

		return self.__observe()

	def render(self, mode: str = "human", debug_dict: Optional[Dict[str, Any]] = None) -> None:
		contract("human" == mode, "Currently, only \"human\" rendering mode is supported, got mode \"{0}\"!", mode)

		if self.renderer is None:
			self.renderer = DeepSeaTreasureV0Renderer(self.theme, 48, 48, self.seabed.shape[0], int(np.max(self.seabed) + 1))	# type: ignore[no-untyped-call]

		contract(self.renderer is not None, "Failed to create {0}.", DeepSeaTreasureV0Renderer.__class__.__name__)

		self.renderer.render(
			submarines=[(int(self.sub_pos[0]), int(self.sub_pos[1]))],
			treasure_values=self.treasures,
			seabed=self.seabed,
			debug_info=debug_dict,
			render_grid=self.render_grid,
			render_treasure_values=self.render_treasure_values
		)

	def config(self) -> Dict[str, Any]:
		acceleration_start_index: int = (int(self.acceleration_levels.shape[0]) // 2) + 1

		treasure_values: List[List[Union[List[int], float]]] = []

		for x in range(int(self.seabed.shape[0])):
			y: int = int(self.seabed[x])

			treasure_values.append([[x, y], self.treasures[(x, y)]])

		return {
			"acceleration_levels": [int(i) for i in self.acceleration_levels[acceleration_start_index:]],
			"implicit_collision_constraint": bool(self.implicit_collision_objective),
			"max_steps": int(self.max_time_steps),
			"max_velocity": float(self.max_vel[0]),
			"treasure_values": treasure_values,
			"render_grid": bool(self.render_grid),
			"render_treasure_values": bool(self.render_treasure_values),
			"theme": self.theme
		}

	@staticmethod
	def new(
			treasure_values: Optional[List[List[Union[List[int], float]]]] = None,
			acceleration_levels: Optional[List[int]] = None,
			implicit_collision_constraint: bool = False,
			max_steps: int = 1000,
			max_velocity: float = 1.0,
			render_grid: bool = False,
			render_treasure_values: bool = False,
			theme: Theme = Theme.default()
	) -> DeepSeaTreasureV0:
		default_treasures: List[List[Union[List[int], float]]] = [
			[[0, 1], 1.0],
			[[1, 2], 2.0],
			[[2, 3], 3.0],
			[[3, 4], 5.0],
			[[4, 4], 8.0],
			[[5, 4], 16.0],
			[[6, 7], 24.0],
			[[7, 7], 50.0],
			[[8, 9], 74.0],
			[[9, 10], 124.0]
		]
		default_acceleration_levels: List[int] = [1, 2, 3]

		treasures: List[List[Union[List[int], float]]] = treasure_values if (treasure_values is not None) else default_treasures

		config = {
			"acceleration_levels": acceleration_levels if (
						acceleration_levels is not None) else default_acceleration_levels,
			"implicit_collision_constraint": int(implicit_collision_constraint),
			"max_steps": max_steps,
			"max_velocity": max_velocity,
			"treasure_values": treasures,
			"render_grid": int(render_grid),
			"render_treasure_values": int(render_treasure_values),
			"theme": theme
		}

		return DeepSeaTreasureV0(config)

	@staticmethod
	def schema() -> Dict[str, Any]:
		schema: Dict[str, Any] = json.loads(pkg_resources.read_text("deep_sea_treasure.schema",
																	"deep_sea_treasure.schema.json"))
		return schema
