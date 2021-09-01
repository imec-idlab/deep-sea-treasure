# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gym
import unittest

import jsonschema
import numpy as np

from deep_sea_treasure import DeepSeaTreasureV0, VamplewWrapper

from jsonschema import ValidationError


class VamplewWrapperTest(unittest.TestCase):
	def setUp(self) -> None:
		self.dst = DeepSeaTreasureV0.new()

	def test_empty_config(self) -> None:
		self.assertRaises(ValidationError, VamplewWrapper, self.dst, {})

	def test_invalid_idle_enable(self) -> None:
		self.assertRaises(ValidationError, VamplewWrapper, self.dst, {"enable_idle": 2})

	def test_spaces(self):
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst, enable_idle=True)

		self.assertEqual(vmplw.observation_space.shape, (2,))
		self.assertEqual(vmplw.reward_space.shape, (2,))
		self.assertEqual(vmplw.action_space.n, 5)

	def test_debug(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst, enable_idle=True)

		_, _, _, debug = vmplw.step(np.asarray([0, 0, 0, 0, 1]))

		self.assertIn("env", debug)
		self.assertIn("inner", debug)
		self.assertIn("idle_enabled", debug)
		self.assertIn("last_velocity", debug)
		self.assertIn("desired_velocity", debug)

		self.assertEqual("VamplewWrapper", debug["env"])
		self.assertEqual(True, debug["idle_enabled"])
		np.testing.assert_equal(debug["last_velocity"], np.asarray([0, 0]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([0, 0]))

	def test_move_normal(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst)

		obs, reward, _, debug = vmplw.step(np.asarray([0, 1, 0, 0]))

		self.assertEqual((2,), obs.shape)
		self.assertEqual((2,), reward.shape)

		np.testing.assert_equal(debug["last_velocity"], np.asarray([0, 0]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([1, 0]))

		np.testing.assert_equal(obs, np.asarray([0, 1]))

		obs, _, _, debug = vmplw.step(np.asarray([0, 1, 0, 0]))

		np.testing.assert_equal(debug["last_velocity"], np.asarray([1, 0]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([1, 0]))

		np.testing.assert_equal(obs, np.asarray([0, 2]))

		obs, _, _, debug = vmplw.step(np.asarray([0, 0, 1, 0]))

		np.testing.assert_equal(debug["last_velocity"], np.asarray([1, 0]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([0, 1]))

		np.testing.assert_equal(obs, np.asarray([1, 2]))

		obs, _, _, debug = vmplw.step(np.asarray([0, 0, 0, 1]))

		np.testing.assert_equal(debug["last_velocity"], np.asarray([0, 1]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([-1, 0]))

		np.testing.assert_equal(obs, np.asarray([1, 1]))

	def test_up_collision(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst)

		obs, _, _, debug = vmplw.step(np.asarray([1, 0, 0, 0]))

		np.testing.assert_equal(debug["last_velocity"], np.asarray([0, 0]))
		np.testing.assert_equal(debug["desired_velocity"], np.asarray([0, -1]))

		np.testing.assert_equal(obs, np.asarray([0, 0]))

	def test_reward_space(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst)

		self.assertIsInstance(vmplw.reward_space, gym.spaces.Box)

		space: gym.spaces.Box = vmplw.reward_space

		self.assertEqual(np.float32, space.dtype)
		self.assertEqual((2,), space.shape)
		np.testing.assert_equal(np.asarray([0, -1.0]), space.low)
		np.testing.assert_equal(np.asarray([124.0, -1.0]), space.high)

	def test_reset(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst, enable_idle=True)

		obs, _, _, debug = vmplw.step(np.asarray([0, 1, 0, 0, 0]))

		np.testing.assert_equal(vmplw.last_velocity, np.asarray([1, 0]))

		reset_obs = vmplw.reset()

		self.assertEqual((2,), reset_obs.shape)
		np.testing.assert_equal(np.asarray([0, 0]), reset_obs)
		np.testing.assert_equal(vmplw.last_velocity, np.asarray([0, 0]))

	def test_integer_step_movement(self) -> None:
		vmplw: VamplewWrapper = VamplewWrapper.new(self.dst)

		exp_obs_1: np.ndarray = np.asarray(
			[0, 1]
		)

		# Velocity: x: +2, y: -1
		obs_1, rew_1, done_1, info_1 = vmplw.step(1)

		self.assertFalse(done_1)
		self.assertEqual(exp_obs_1.shape, obs_1.shape)
		np.testing.assert_equal(exp_obs_1, obs_1)

	def test_config(self) -> None:
		wrapper: VamplewWrapper = VamplewWrapper.new(self.dst)

		config = wrapper.config()

		# First, make sure that the result of `.config()` validates against the schema
		validator = jsonschema.Draft7Validator(schema=VamplewWrapper.schema())

		# Test will fail if this raises a ValidationException
		validator.validate(config)

		# Next, make sure we can construct a new environment from this config
		new_dst: VamplewWrapper = VamplewWrapper(self.dst, config)

		self.assertIn("inner", config)
		self.assertEqual(False, config["enable_idle"])
