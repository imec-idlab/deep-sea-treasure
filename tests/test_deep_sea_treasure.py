# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import gym
import unittest

from typing import List, Union

from deep_sea_treasure import DeepSeaTreasureV0

import numpy as np
from jsonschema import ValidationError


class DeepSeaTreasureV0Test(unittest.TestCase):
	def test_empty_config(self) -> None:
		config = {}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_no_treasures_config(self) -> None:
		config = {
			"treasure_values": [],
			"acceleration_levels": [1.0],
			"max_steps": 1,
			"max_velocity": 1.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_no_acceleration_levels_config(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [],
			"max_steps": 1,
			"max_velocity": 1.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_single_treasure_config(self) -> None:
		config = {
			"treasure_values": [[1, 1]],
			"acceleration_levels": [1.0],
			"max_steps": 1,
			"max_velocity": 1.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_default_config(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [1.0, 2.0, 3.0],
			"implicit_collision_constraint": 0,
			"max_steps": 100,
			"max_velocity": 5.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0(env_config=config)

	def test_negative_acceleration(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [-1.0],
			"max_steps": 100,
			"max_velocity": 5.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_negative_steps(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [1.0],
			"max_steps": -1,
			"max_velocity": 5.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_negative_max_velocity(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [1.0],
			"max_steps": 1,
			"max_velocity": -5.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		self.assertRaises(ValidationError, DeepSeaTreasureV0, config)

	def test_new_default_config(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new()

		expected_observation: np.ndarray = np.asarray(
			[[0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
			[0, 1, 2, 3, 4, 4, 4, 7, 7, 9, 10]]
		)

		actual_observation: np.ndarray = dst.reset()

		self.assertEqual(expected_observation.shape, actual_observation.shape)

		for (exp, act) in zip(np.nditer(expected_observation), np.nditer(actual_observation)):
			self.assertEqual(exp, act)

	def test_step_movement(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2], max_velocity=2)

		exp_obs_1: np.ndarray = np.asarray(
			[[2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
			[1, 0, 1, 2, 3, 3, 3, 6, 6, 8, 9]]
		)

		# Velocity: x: +2, y: -1
		obs_1, rew_1, done_1, info_1 = dst.step((
			np.asarray([0, 0, 0, 0, 1]),	# x + 2
			np.asarray([0, 0, 0, 1, 0])	# y + 1
		))

		self.assertFalse(done_1)
		self.assertEqual(exp_obs_1.shape, obs_1.shape)

		np.testing.assert_equal(exp_obs_1, obs_1)

		exp_obs_2: np.ndarray = np.asarray(
			[[1, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
			 [0, 0, 1, 2, 3, 3, 3, 6, 6, 8, 9]]
		)

		# Velocity: x: +1, y: 0
		obs_2, rew_2, done_2, info_2 = dst.step((
			np.asarray([0, 1, 0, 0, 0]),	# x - 1
			np.asarray([0, 1, 0, 0, 0])	# y - 1
		))

		self.assertFalse(done_2)
		self.assertEqual(exp_obs_2.shape, obs_2.shape)

		for i, (exp, act) in enumerate(zip(np.nditer(exp_obs_2), np.nditer(obs_2))):
			self.assertEqual(exp, act, f"Mismatch at element {i} ({exp} != {act})")

		exp_obs_3: np.ndarray = np.asarray(
			[[1, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
			 [0, 0, 1, 2, 3, 3, 3, 6, 6, 8, 9]]
		)

		# Velocity: x: +1, y: 0
		obs_3, rew_3, done_3, info_3 = dst.step((
			np.asarray([0, 0, 1, 0, 0]),	# x + 0
			np.asarray([0, 0, 1, 0, 0])	# y + 0
		))

		self.assertFalse(done_3)
		self.assertEqual(exp_obs_3.shape, obs_3.shape)

		for i, (exp, act) in enumerate(zip(np.nditer(exp_obs_3), np.nditer(obs_3))):
			self.assertEqual(exp, act, f"Mismatch at element {i} ({exp} != {act})")

	def test_high_velocity(self) -> None:
		config = {
			"treasure_values": [
				[[0, 1], 1],
				[[1, 2], 2],
				[[2, 3], 3],
				[[3, 4], 5],
				[[4, 4], 8],
				[[5, 4], 16],
				[[6, 7], 24],
				[[7, 7], 50],
				[[8, 9], 74],
				[[9, 10], 124]
			],
			"acceleration_levels": [1.0],
			"implicit_collision_constraint": 0,
			"max_steps": 100,
			"max_velocity": 2.0,
			"render_grid": 0,
			"render_treasure_values": 0,
			"theme": "default"
		}

		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0(env_config=config)

		obs, _, _, debug = dst.step((np.asarray([0, 0, 1]), np.asarray([0, 1, 0])))

		# We increased x-velocity by 1
		self.assertEqual(obs[0][0], 1.0)
		self.assertEqual(obs[1][0], 0.0)
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])
		self.assertEqual(2, debug["velocity"]["max_x"])
		self.assertEqual(2, debug["velocity"]["max_y"])
		self.assertEqual(-2, debug["velocity"]["min_x"])
		self.assertEqual(-2, debug["velocity"]["min_y"])

		obs, _, _, debug = dst.step((np.asarray([0, 0, 1]), np.asarray([0, 1, 0])))

		# We increased x-velocity by 1
		self.assertEqual(obs[0][0], 2.0)
		self.assertEqual(obs[1][0], 0.0)
		self.assertEqual(2, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])
		self.assertEqual(2, debug["velocity"]["max_x"])
		self.assertEqual(2, debug["velocity"]["max_y"])
		self.assertEqual(-2, debug["velocity"]["min_x"])
		self.assertEqual(-2, debug["velocity"]["min_y"])

		obs, _, _, debug = dst.step((np.asarray([0, 0, 1]), np.asarray([0, 1, 0])))

		# We can no longer increase x-velocity, since we are already at maximum velocity
		self.assertEqual(obs[0][0], 2.0)
		self.assertEqual(obs[1][0], 0.0)
		self.assertEqual(2, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])
		self.assertEqual(2, debug["velocity"]["max_x"])
		self.assertEqual(2, debug["velocity"]["max_y"])
		self.assertEqual(-2, debug["velocity"]["min_x"])
		self.assertEqual(-2, debug["velocity"]["min_y"])

	def test_clip_out_of_bounds_x(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2])

		# Try to move out-of-bounds in the x-direction near the origin
		obs, rew, _, debug = dst.step((np.asarray([0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		# Check debug collision
		self.assertTrue(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Check reward
		np.testing.assert_equal(np.asarray([0.0, -1.0]), rew)

	def test_clip_out_of_bounds_y(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2])

		# Try to move out-of-bounds in the y-direction near the origin
		obs, rew, _, debug = dst.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertTrue(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Check reward
		np.testing.assert_equal(np.asarray([0.0, -1.0]), rew)

	def test_clip_seabed_perfect_diagonal(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2])

		# Move down 1 space, so we are hovering right above the seabed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(1.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(1, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Try to clip the seabed (velocity is now (+1, +1))
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 1, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertTrue(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# After clipping, try to move in a way that's allowed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 1, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(1.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-1.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(1, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

	def test_clip_seabed_knight_move(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2])

		# Move down 1 space, so we are hovering right above the seabed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(1.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(1, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Try to clip the seabed, on a non-obvious square (2 to the side, 1 down) (velocity is now (+2, +1)
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 1]), np.asarray([0, 0, 0, 1, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(0.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(0, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertTrue(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# After clipping, try to move in a way that's allowed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 1, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(1.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-1.0, obs[0][1])
		self.assertEqual(0.0, obs[1][1])
		self.assertEqual(1, debug["position"]["x"])
		self.assertEqual(1, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

	def test_clip_seabed_left_fast(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2, 3], max_velocity=5)

		# Move right 1 space
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(1.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-1.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(1, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Move down 2 spaces
		obs, _, _, debug = dst.step((np.asarray([0, 0, 1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 0, 0, 1, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(2.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(2, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-1.0, obs[0][1])
		self.assertEqual(-1.0, obs[1][1])
		self.assertEqual(1, debug["position"]["x"])
		self.assertEqual(2, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Slam into the seabed at full speed
		obs, _, _, debug = dst.step((np.asarray([1, 0, 0, 0, 0, 0, 0]), np.asarray([0, 1, 0, 0, 0, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-1.0, obs[0][1])
		self.assertEqual(-1.0, obs[1][1])
		self.assertEqual(1, debug["position"]["x"])
		self.assertEqual(2, debug["position"]["y"])

		# Check debug collision
		self.assertTrue(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# After clipping, try to move in a way that's allowed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(1.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-2.0, obs[0][1])
		self.assertEqual(-1.0, obs[1][1])
		self.assertEqual(2, debug["position"]["x"])
		self.assertEqual(2, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

	def test_rewards(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2, 3], max_velocity=3)

		# Idle, check rewards for idling
		obs, rewards, _, _ = dst.step((np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		self.assertEqual((2,), rewards.shape)
		self.assertEqual(0.0, rewards[0])
		self.assertEqual(-1.0, rewards[1])

		# Move into first treasure, check rewards for hitting small treasure
		obs, rewards, _, _ = dst.step((np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1, 0, 0])))

		self.assertEqual(1.0, rewards[0])
		self.assertEqual(-1.0, rewards[1])

		# Book it to the furthest away treasure, see if we also get the correct value there
		# Slam into the side of the world as hard as we can
		obs, rewards, _, _ = dst.step((np.asarray([0, 0, 0, 0, 0, 0, 1]), np.asarray([0, 0, 1, 0, 0, 0, 0])))

		# First, move right (constant velocity of 3 means that we get there in exactly 2 more steps)
		while 0.0 < obs[0][-1]:
			self.assertEqual(0.0, rewards[0])
			self.assertEqual(-1.0, rewards[1])

			self.assertEqual(3.0, obs[0][0])
			self.assertEqual(0.0, obs[1][0])

			obs, rewards, _, _ = dst.step((np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		obs, rewards, _, _ = dst.step((np.asarray([1, 0, 0, 0, 0, 0, 0]), np.asarray([0, 0, 0, 0, 0, 0, 1])))

		# Then, move down
		# First, move right (constant velocity of 3 means that we get there in exactly 2 more steps)
		while 0.0 < obs[1][-1:]:
			self.assertEqual(0.0, rewards[0])
			self.assertEqual(-1.0, rewards[1])

			self.assertEqual(0.0, obs[0][0])
			self.assertEqual(3.0, obs[1][0])

			obs, rewards, _, _ = dst.step((np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		self.assertEqual(124.0, rewards[0])
		self.assertEqual(-1.0, rewards[1])

	def test_max_steps_idle(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(max_steps=3, acceleration_levels=[1, 2])

		for i in range(2):
			_, _, done, debug = dst.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

			self.assertFalse(done)
			self.assertEqual(i + 1, debug["time"]["current"])
			self.assertEqual(3, debug["time"]["max"])

		_, _, done, debug = dst.step((np.asarray([0, 0, 1, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		self.assertTrue(done)
		self.assertEqual(3, debug["time"]["current"])
		self.assertEqual(3, debug["time"]["max"])

	def test_diagonal_collision(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2], max_velocity=2)

		# Move 2 squares right
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 1]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(2.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(2, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-2.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(2, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Stop all movement
		obs, _, _, debug = dst.step((np.asarray([1, 0, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-2.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(2, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Induce diagonal collision
		obs, _, _, debug = dst.step((np.asarray([1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-2.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(2, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertTrue(debug["collision"]["diagonal"])

		# After clipping, try to move in a way that's allowed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 1, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check velocity
		self.assertEqual(1.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(1, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		# Check position
		self.assertEqual(-3.0, obs[0][1])
		self.assertEqual(1.0, obs[1][1])
		self.assertEqual(3, debug["position"]["x"])
		self.assertEqual(0, debug["position"]["y"])

		# Check debug collision
		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

	def test_right_collision(self) -> None:
		treasure_values: List[List[Union[List[int], float]]] = [
			[[0, 1], 1.0],
			[[1, 2], 2.0],
			[[2, 3], 3.0],
			[[3, 3], 5.0],
			[[4, 2], 8.0],
			[[5, 1], 16.0],
			[[6, 7], 24.0],
			[[7, 7], 50.0],
			[[8, 9], 74.0],
			[[9, 10], 124.0]
		]

		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(treasure_values=treasure_values, acceleration_levels=[1, 2, 3], max_velocity=5)

		# Move 3 squares right
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 0, 0, 1]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(3.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(3, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Move another 3 squares right, we are now right up against a wall
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 0, 0, 1]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(5.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(5, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		self.assertFalse(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

		# Move 3 squares right, crashing into the seabed
		obs, _, _, debug = dst.step((np.asarray([0, 0, 0, 0, 0, 0, 1]), np.asarray([0, 0, 0, 1, 0, 0, 0])))

		# Check velocity
		self.assertEqual(0.0, obs[0][0])
		self.assertEqual(0.0, obs[1][0])
		self.assertEqual(0, debug["velocity"]["x"])
		self.assertEqual(0, debug["velocity"]["y"])

		self.assertTrue(debug["collision"]["horizontal"])
		self.assertFalse(debug["collision"]["vertical"])
		self.assertFalse(debug["collision"]["diagonal"])

	def test_reward_space(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new()

		self.assertIsInstance(dst.reward_space, gym.spaces.Box)

		space: gym.spaces.Box = dst.reward_space

		self.assertEqual(np.float32, space.dtype)
		self.assertEqual((2,), space.shape)
		np.testing.assert_equal(np.asarray([0, -1.0]), space.low)
		np.testing.assert_equal(np.asarray([124.0, -1.0]), space.high)

	def test_reset(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new()

		reset_obs = dst.reset()

		self.assertEqual((2, 11), reset_obs.shape)

	def test_integer_step_movement(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(acceleration_levels=[1, 2, 3], max_velocity=2)

		exp_obs_1: np.ndarray = np.asarray(
			[[2, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
			[1, 0, 1, 2, 3, 3, 3, 6, 6, 8, 9]]
		)

		# Velocity: x: +2, y: -1
		obs_1, rew_1, done_1, info_1 = dst.step((5,	4))

		self.assertFalse(done_1)
		self.assertEqual(exp_obs_1.shape, obs_1.shape)
		np.testing.assert_equal(exp_obs_1, obs_1)

	def test_clip_out_of_bounds_implicit_collision_reward(self) -> None:
		dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(implicit_collision_constraint=True, acceleration_levels=[1, 2])

		np.testing.assert_equal(np.asarray([-1.0, -2.0]), dst.reward_space.low)

		# Try to move out-of-bounds in the x-direction near the origin
		obs, rew, _, debug = dst.step((np.asarray([0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0])))

		# Check reward
		np.testing.assert_equal(np.asarray([-1.0, -2.0]), rew)
