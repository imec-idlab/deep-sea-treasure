# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gym
import unittest
import numpy as np

from deep_sea_treasure import DeepSeaTreasureV0, FuelWrapper

from jsonschema import ValidationError


class FuelWrapperTest(unittest.TestCase):
	dst: DeepSeaTreasureV0

	def setUp(self) -> None:
		self.dst = DeepSeaTreasureV0.new(max_velocity=3, acceleration_levels=[1, 2])

	def test_empty_config(self) -> None:
		self.assertRaises(ValidationError, FuelWrapper, self.dst, {})

	def test_single_fuel_cost(self) -> None:
		self.assertRaises(AssertionError, FuelWrapper, self.dst, {"fuel_cost": [1]})

	def test_correct_config(self) -> None:
		wrapper: FuelWrapper = FuelWrapper(self.dst, {"fuel_cost": [1, 2]})

	def test_no_move(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		_, rew, _, debug = wrapper.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		self.assertEqual((3,), rew.shape)
		self.assertEqual(0, rew[-1])
		self.assertIn("inner", debug)
		self.assertEqual(2, debug["action_space_midpoint"])

	def test_fast_move_symmetry(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		_, rew, _, debug = wrapper.step((np.asarray([0, 0, 0, 0, 1]), np.asarray([0, 0, 1, 0, 0])))

		self.assertEqual(-2, rew[-1])
		self.assertEqual(-2, debug["fuel_cost"]["x"])
		self.assertEqual(0, debug["fuel_cost"]["y"])

		_, rew, _, debug = wrapper.step((np.asarray([1, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		self.assertEqual(-2, rew[-1])
		self.assertEqual(-2, debug["fuel_cost"]["x"])
		self.assertEqual(0, debug["fuel_cost"]["y"])

	def test_collision_reward(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		_, rew, _, debug = wrapper.step((np.asarray([0, 0, 0, 0, 1]), np.asarray([0, 0, 0, 0, 1])))

		# If the agent collides, no fuel was spent
		self.assertEqual(0, rew[-1])
		self.assertEqual(0, debug["fuel_cost"]["x"])
		self.assertEqual(0, debug["fuel_cost"]["y"])

	def test_sum_reward(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		_, rew, _, debug = wrapper.step((np.asarray([0, 0, 0, 1, 0]), np.asarray([0, 0, 0, 1, 0])))

		self.assertEqual(-2, rew[-1])
		self.assertEqual(-1, debug["fuel_cost"]["x"])
		self.assertEqual(-1, debug["fuel_cost"]["y"])

	def test_reward_space(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		self.assertIsInstance(wrapper.reward_space, gym.spaces.Box)

		space: gym.spaces.Box = wrapper.reward_space

		self.assertEqual(np.float32, space.dtype)
		self.assertEqual((3,), space.shape)
		np.testing.assert_equal(np.asarray([0, -1.0, -2]), space.low)
		np.testing.assert_equal(np.asarray([124.0, -1.0, 0]), space.high)

	def test_reset(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		reset_obs = wrapper.reset()

		self.assertEqual((2, 11), reset_obs.shape)

	def test_integer_step_movement(self) -> None:
		wrapper: FuelWrapper = FuelWrapper.new(self.dst, fuel_cost=[1, 2])

		exp_obs_1: np.ndarray = np.asarray(
			[[2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
			[1, 0, 1, 2, 3, 3, 3, 6, 6, 8, 9]]
		)

		# Velocity: x: +2, y: -1
		obs_1, rew_1, done_1, info_1 = wrapper.step((4,	3))

		self.assertFalse(done_1)
		self.assertEqual(exp_obs_1.shape, obs_1.shape)
		np.testing.assert_equal(exp_obs_1, obs_1)
