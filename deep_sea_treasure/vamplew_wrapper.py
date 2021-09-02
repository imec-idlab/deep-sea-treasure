# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

import copy
import gym
import json
from jsonschema import Draft7Validator
import numpy as np
import numpy.typing as npt
import importlib.resources as pkg_resources
from typing import Any, cast, Dict, List, Tuple, Union, Optional

from deep_sea_treasure.contract import contract


class VamplewWrapper(gym.Wrapper): #type: ignore[misc]
	idle_enabled: bool

	last_velocity: npt.NDArray[np.int32]
	action_to_velocity_matrix: npt.NDArray[np.int32]

	def __init__(self, env: gym.Env, wrapper_config: Dict[str, Any]):
		super(VamplewWrapper, self).__init__(env)

		contract(hasattr(env, "config"), f"{self.__class__.__name__} was used to wrap {env.__class__.__name__} which doesn't have a \"config\" attribute required for extracting configuration information!")

		# Validate own config
		Draft7Validator(schema=VamplewWrapper.schema()).validate(wrapper_config)

		self.idle_enabled = bool(wrapper_config["enable_idle"])
		self.last_velocity = np.zeros(shape=(2,), dtype=np.int32)

		self.reward_space = self.env.reward_space
		self.action_space = gym.spaces.Discrete(n=4 + int(wrapper_config["enable_idle"]))
		self.observation_space = gym.spaces.Box(low=float("-inf"), high=float("+inf"), shape=(2,))

		# Validate properties of wrapped environment, establishing post-conditions
		contract(isinstance(self.env.action_space, gym.spaces.Tuple), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}!", type(self.env.action_space).__name__)
		contract(2 == len(self.env.action_space.spaces), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}({1})!", type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))
		contract(self.env.action_space.spaces[0].n == self.env.action_space.spaces[1].n, "x- and y- action space must have the same size, got {0} and {1}!", self.env.action_space.spaces[0].n, self.env.action_space.spaces[1].n)

		for i, space in enumerate(self.env.action_space.spaces):
			contract(isinstance(space, gym.spaces.Discrete), "Wrapped environment action space must be a Tuple of 2 Discrete spaces, got {0}({1})!", type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))
			contract(5 <= space.n, "Inner Discrete spaces must have at least {0} actions, got {1} actions for inner space at index {2} of type {3}({4})!", 5, space.n, i, type(self.env.action_space).__name__, ', '.join([type(t_space).__name__ for t_space in self.env.action_space.spaces]))

		self.action_to_velocity_matrix = np.asarray(
			[
				[0, -1],  # UP
				[1, 0],  # RIGHT
				[0, 1],  # DOWN
				[-1, 0],  # LEFT
				[0, 0]  # IDLE
			],
			dtype=np.int32
		)

		self.reset()

	def step(self, action: Union[int, float, npt.NDArray[np.int32]]) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.single], bool, Dict[str, Any]]:
		"""
		Actions are encoded as [UP, RIGHT, DOWN, LEFT (, IDLE)]
		:param action:
		:return:
		"""
		if not isinstance(action, np.ndarray):
			contract((0 <= int(action)) and (int(action) < self.action_space.n), "Integer action must be in range [{0}, {1}[", 0, self.action_space.n)

			index = int(action)
			action = np.zeros(shape=(self.action_space.n,), dtype=np.int32)
			action[index] = 1

		contract(isinstance(action, np.ndarray), "Action must be {0} after action normalization, got {1}!", np.ndarray.__name__, type(action))

		contract(action.shape == (self.action_space.n,), "Action space must be a 1-hot encoding with {0} actions, got action with shape {1}.", self.action_space.n, action.shape)
		contract(1 == int(np.sum(action)), "Action space must be a 1-hot encoding with {0} actions, got {1}.", self.action_space.n, action)

		if not self.idle_enabled:
			contract((4,) == action.shape, "Action must not include idle action if idle is disabled!")

			action = np.append(action, 0)	# type: ignore[no-untyped-call]

		contract(isinstance(action, np.ndarray), "Action must be {0} after action normalization!", np.ndarray.__name__)
		action = cast(npt.NDArray[np.int32], action)

		contract((5,) == action.shape, "Action had incorrect dimensions ({0}) after pre-processing, expected {1}!", action.shape, (5,))

		desired_velocity: npt.NDArray[np.int32] = np.matmul(np.expand_dims(action, axis=0), self.action_to_velocity_matrix).transpose().squeeze()	# type: ignore[no-untyped-call]

		contract((2,) == desired_velocity.shape, "Conversion from action to desired velocity resulted in matrix with unexpected size, expected {0}, got {1}", (2,), desired_velocity.shape)

		dv: npt.NDArray[np.int32] = desired_velocity - self.last_velocity

		x_action = np.zeros(shape=(self.env.action_space.spaces[0].n,))
		y_action = np.zeros(shape=(self.env.action_space.spaces[1].n,))

		x_action[(x_action.shape[0] // 2) + int(dv[0])] = 1
		y_action[(y_action.shape[0] // 2) + int(dv[1])] = 1

		obs, rew, done, debug = self.env.step((x_action, y_action))

		inner_debug = debug

		while inner_debug["env"] != "DeepSeaTreasureV0":
			contract("inner" in inner_debug, f"`debug` must contain the wrapped debug dictionary in the `inner` attribute.")
			inner_debug = inner_debug["inner"]

		new_obs: List[int] = [
			int(inner_debug["position"]["y"]),
			int(inner_debug["position"]["x"])
		]

		debug_info: Dict[str, Any] = {
			"env": self.__class__.__name__,
			"inner": debug,
			"idle_enabled": self.idle_enabled,
			"last_velocity": list(self.last_velocity),
			"desired_velocity": list(desired_velocity)
		}

		self.last_velocity = copy.deepcopy(obs[:, 0])

		return np.asarray(new_obs, dtype=np.int32), rew, done, debug_info

	def reset(self, **kwargs: None) -> npt.NDArray[np.single]:
		super(VamplewWrapper, self).reset()
		self.last_velocity = np.zeros_like(self.last_velocity)
		return np.zeros(shape=(2,), dtype=np.single)

	def render(self, mode: str = "human", **kwargs: None) -> None:
		super().render(mode, **kwargs)

	@staticmethod
	def new(env: gym.Env, enable_idle: Optional[bool] = None) -> VamplewWrapper:

		config = VamplewWrapper.default_config()

		if enable_idle is not None:
			config["enable_idle"] = enable_idle

		return VamplewWrapper(env, config)

	@staticmethod
	def default_config() -> Dict[str, Any]:
		return {
			"enable_idle": False
		}

	@staticmethod
	def schema() -> Dict[str, Any]:
		schema: Dict[str, Any] = json.loads(pkg_resources.read_text("deep_sea_treasure.schema", "vamplew_wrapper.schema.json"))

		return schema

	def config(self) -> Dict[str, Any]:
		return {
			"enable_idle": bool(self.idle_enabled),
			"inner": self.env.config()
		}
