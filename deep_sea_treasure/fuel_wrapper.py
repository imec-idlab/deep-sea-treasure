# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import gym
import json
import numpy as np
import numpy.typing as npt

import importlib.resources as pkg_resources

from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema import Draft7Validator

from deep_sea_treasure.contract import contract


class FuelWrapper(gym.Wrapper): #type: ignore[misc]
	action_space_mid_point: int
	fuel_cost: npt.NDArray[np.int32]

	def __init__(self, env: gym.Env, wrapper_config: Dict[str, Any]):
		super(FuelWrapper, self).__init__(env)

		contract(hasattr(env, "config"), f"{self.__class__.__name__} was used to wrap {env.__class__.__name__} which doesn't have a \"config\" attribute required for extracting configuration information!")

		# Validate own config
		Draft7Validator(schema=FuelWrapper.schema()).validate(wrapper_config)

		inner_low = self.env.reward_space.low
		inner_high = self.env.reward_space.high

		self.reward_space = gym.spaces.Box(
			low=np.append(inner_low, -max(wrapper_config["fuel_cost"])),	# type: ignore[no-untyped-call]
			high=np.append(inner_high, 0),									# type: ignore[no-untyped-call]
			shape=(self.env.reward_space.shape[0] + 1,),
			dtype=np.float32
		)

		self.action_space = self.env.action_space
		self.observation_space = self.env.observation_space

		self.fuel_cost = np.asarray(wrapper_config["fuel_cost"], dtype=np.int32)
		self.action_space_mid_point = len(self.fuel_cost)

		# Validate properties of wrapped environment, establishing post-conditions
		contract(isinstance(self.env.action_space, gym.spaces.Tuple), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}!", type(self.env.action_space).__name__)
		contract(2 == len(self.env.action_space.spaces), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}({1})!", type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))

		for i, space in enumerate(self.env.action_space.spaces):
			contract(isinstance(space, gym.spaces.Discrete), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}({1})!", type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))
			contract(((2 * len(self.fuel_cost)) + 1) == space.n, "Inner Discrete spaces must have {0} actions, given {1} different fuel costs, got {2} actions for inner space at index {3} of type {4}({5})!", (2 * len(self.fuel_cost)) + 1, len(self.fuel_cost), space.n, i, type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))

		self.reset()

	def step(self, action: Union[Tuple[int, int], Tuple[float, float], Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]]) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.single], bool, Dict[str, Any]]:
		x_action, y_action = action

		if not isinstance(x_action, np.ndarray):
			contract((0 <= int(x_action)) and (int(x_action) < self.action_space[0].n), "Integer action must be in range [{0}, {1}[", 0, self.action_space[0].n)

			index = int(x_action)
			x_action = np.zeros(shape=(self.action_space[0].n,), dtype=np.int32)
			x_action[index] = 1

		if not isinstance(y_action, np.ndarray):
			contract((0 <= int(y_action)) and (int(y_action) < self.action_space[1].n), "Integer action must be in range [{0}, {1}[", 0, self.action_space[1].n)

			index = int(y_action)
			y_action = np.zeros(shape=(self.action_space[1].n,), dtype=np.int32)
			y_action[index] = 1

		contract(isinstance(x_action, np.ndarray), "X-action must be {0} after action normalization, got {1}!", np.ndarray.__name__, type(x_action))
		contract(isinstance(y_action, np.ndarray), "Y-action must be {0} after action normalization, got {1}!", np.ndarray.__name__, type(y_action))

		contract(not bool(np.any(np.isnan(x_action))), "X-Action contained NaNs: {0}!", x_action)
		contract(not bool(np.any(np.isnan(y_action))), "Y-Action contained NaNs: {0}!", y_action)

		x_fuel_index: int = abs(int(np.argmax(x_action)) - self.action_space_mid_point)
		y_fuel_index: int = abs(int(np.argmax(y_action)) - self.action_space_mid_point)

		x_fuel_cost: int = -self.fuel_cost[x_fuel_index - 1] if (0 < x_fuel_index) else 0
		y_fuel_cost: int = -self.fuel_cost[y_fuel_index - 1] if (0 < y_fuel_index) else 0

		obs, rew, done, debug = self.env.step(action)

		new_rewards: npt.NDArray[np.single] = np.zeros(shape=(len(rew) + 1,), dtype=np.single)
		new_rewards[:len(rew)] = rew

		contract("collision" in debug, "Debug dictionary from inner environment was missing \"collision\" field! (Dictionary: {0})", debug)

		new_debug: Dict[str, Any] = {
			"env": self.__class__.__name__,
			"action_space_midpoint": self.action_space_mid_point,
			"fuel_cost": {
				"x": 0,
				"y": 0
			},
			"inner": debug
		}

		if not (debug["collision"]["horizontal"] or debug["collision"]["vertical"] or debug["collision"]["diagonal"]):
			new_rewards[-1] = int(x_fuel_cost + y_fuel_cost)
			new_debug["fuel_cost"]["x"] = x_fuel_cost
			new_debug["fuel_cost"]["y"] = y_fuel_cost

		return obs, new_rewards, done, new_debug

	def reset(self, **kwargs: None) -> npt.NDArray[np.int32]:
		return self.env.reset(**kwargs)	#type: ignore[no-any-return]

	def render(self, mode: str = "human", **kwargs: None) -> None:
		super().render(mode, **kwargs)

	@staticmethod
	def new(env: gym.Env, fuel_cost: Optional[List[int]] = None) -> FuelWrapper:
		config: Dict[str, Any] = FuelWrapper.default_config()

		if fuel_cost is not None:
			config["fuel_cost"] = list(fuel_cost)

		return FuelWrapper(env, config)

	@staticmethod
	def default_config() -> Dict[str, Any]:
		return {
			"fuel_cost": [1, 4, 9]
		}

	@staticmethod
	def schema() -> Dict[str, Any]:
		schema: Dict[str, Any] = json.loads(pkg_resources.read_text("deep_sea_treasure.schema",
																	"fuel_wrapper.schema.json"))
		return schema

	def config(self) -> Dict[str, Any]:
		return {
			"fuel_cost": [int(i) for i in self.fuel_cost],
			"inner": self.env.config()
		}
