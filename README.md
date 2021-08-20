# DeepSeaTreasure Environment
 
![animation of submarine search for treasure](/res/Deep-Sea-Treasure-v0.gif "DeepSeaTreasure v0")

## Installation
In order to get started with this environment, you can install it using the following command:
```shell
python3 -m pip install  deep_sea_treasure --upgrade
```

## Data
If you are only interested in obtaining the Pareto-front data, you can find that  on the [data/](data/) folder:

- [2-objective Pareto-front](data/2-objective.json)
- [3-objective Pareto-front](data/3-objective.json)

## Example
After installing the environment, you can get started using it like this:
```python
import pygame
import numpy as np
import time

import deep_sea_treasure
from deep_sea_treasure import DeepSeaTreasureV0

# Make sure experiment are reproducible, so people can use the exact same versions
print(f"Using DST {deep_sea_treasure.__version__.VERSION} ({deep_sea_treasure.__version__.COMMIT_HASH})")

dst: DeepSeaTreasureV0 = DeepSeaTreasureV0.new(
	max_steps=1000,
	render_treasure_values=True
)

dst.render()

stop: bool = False
time_reward: int = 0

while not stop:
	events = pygame.event.get()

	action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))

	for event in events:
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				action = (np.asarray([0, 0, 1, 0, 0, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
			elif event.key == pygame.K_RIGHT:
				action = (np.asarray([0, 0, 0, 0, 1, 0, 0]), np.asarray([0, 0, 0, 1, 0, 0, 0]))
			if event.key == pygame.K_UP:
				action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 1, 0, 0, 0, 0]))
			elif event.key == pygame.K_DOWN:
				action = (np.asarray([0, 0, 0, 1, 0, 0, 0]), np.asarray([0, 0, 0, 0, 1, 0, 0]))

			if event.key in {pygame.K_ESCAPE}:
				stop = True

		if event.type == pygame.QUIT:
			stop = True

	_, reward, done, debug_info = dst.step(action)
	time_reward += int(reward[1])

	if done:
		print(f"Found treasure worth {float(reward[0]):5.2f} after {abs(time_reward)} timesteps!")
		time_reward = 0

	if not stop:
		dst.render()
		time.sleep(0.25)

	if done:
		dst.reset()
```

This will allow you to play around in the environment, and move the sub around, finding treasures!

## API
This section will provide you with a detailed description of the API we offer, and how you can use it.
In general, our API matches that of OpenAI's gym package.
This library contains 1 environment (`DeepSeaTreasure-v0`) and 2 wrappers (`FuelWrapper` and `VamplewWrapper`).
In order to aid reproducibility, we include a package version, and source-code commit hash in the python code (`deep_sea_treasure.__version__.VERSION` and `deep_sea_treasure.__version__.COMMIT_HASH`).

The environment we created is a modified version of the environment originally showcased in the paper by [Vamplew et al.](https://doi.org/10.1007/s10994-010-5232-5)
In our version, the agent can move around by providing an acceleration on both the x- and y- direction.
By default, accelerations are discrete, going from -3 to +3. It is the agent's goal to find the largest possible treasure, in the shortest amount of time.
This immediately creates a tricky situation for any agent attempting to learn in this environment,
since the agent has to deal with two conflicting objectives: time and treasure. This is also reflected in the reward we return from the environment,
since this is a 2-element vector (1 element for each objective).

### Wrappers
When wrapping this environment, users should take care to ensure that their `debug_dict` always contains at least an `env` value,
identifying the wrapper that created the dict, and an `inner` value, that contains the debugging information from the wrapped environment.
This is important if you want to use the renderer to render debug information,
since the renderer is only capable of rendering debug information from the core environment itself.

__Example__
```python
from typing import Any, Dict, Tuple

class ExampleWrapper:
	def step(self, action) -> Tuple[Any, Any, bool, Dict[str, Any]]:
		obs, rew, done, debug_dict = self.env.step(action)

		new_debug_dict = {
			"env": self.__class__.__name__	# = "ExampleWrapper"
			"inner": debug_dict
		}

		return obs, rew, done, new_debug_dict
```
### General
All environments and wrappers in this repository provide a `reward_space` attribute,
similar to the `observation_space` and `action_space` attributes normally present in gym environments.
The main purpose behind this is to allow an external observer to determine the number of objectives.


### [`DeepSeaTreasureV0`](deep_sea_treasure/deep_sea_treasure_v0.py)
```python
from deep_sea_treasure import DeepSeaTreasureV0
```
This is the core environment, creation is handled by the `new()` static method.
This is the recommended way of creating environments, since it provides meaningful arguments for changing the various settings for the environment.
Beyond this, there is also a constructor that accepts a dictionary of settings, this was mainly done to retain compatibility with RL frameworks like RLLib.
Internally, the `new()` method simply fills a dictionary, and delegates to the constructor.
The constructor uses JSON schemas along with a couple of extra `assert`s to verify the correctness of a given configuration.
The exact schema can be found in [deep_sea_treasure.schema.json](deep_sea_treasure/schema/deep_sea_treasure.schema.json), and is also returned by `DeepSeaTreasureV0.schema()`.
A copy of the default configuration can be obtained through `DeepSeaTreasureV0.default_config()`.
The table below describes the various options offered by the `new()` method.

| Option                    | Type                                  | Meaning |
| ------------------------- | ------------------------------------- | ------- |
| `treasure_values`         | `List[List[Union[List[int], float]]]` | This option serves two purposes: It shapes the seabed, and determines the value of each treasure. It is an array of `((x, y), treasure)` tuples.
| `acceleration_levels`     | `List[int]`                           | This option provides a list of discrete acceleration levels. Each level should be a positive integer number. Upon creation, the numbers with inverted sign will be used for accelerating left and up, while the numbers themselves signify acceleration either right (x) or down (y).
| `implicit_collision_constraint`	| `bool`						| When this option is enabled, the reward function will be reduced to below its minimum when the submarine causes a collision. This is also reflected in the `reward_space.low`, the values in this array will be 1 below the minimum attainable through normal actions in the environment.
| `max_steps`               | `int`                                 | The environment has 2 ending conditions: 1. The submarine finds a treasure 2. The maximum number of steps is reached, this option sets the maximum allowed number of steps.
| `max_velocity`            | `float`                               | The environment limits the absolute value of the submarine's maximum velocity to this number, to prevent overflow-related physics shenanigans from occurring.
| `render_grid`             | `bool`                                | If this option is enabled, a grid will be shown when rendering the environment, showing the discrete spaces where the submarine can reside.
| `render_treasure_values`  | `bool`                                | If this option is enabled, the value of each treasure will be rendered on top of the treasure, making the rendering slightly clearer.
| `theme`                   | `Theme`                               | We allow the customization of the rendering through the use of the `Theme` class, in conjunction with the `DeepSeaTreasureV0Renderer`, this option allows the user to specify the `Theme` to use when rendering, the default `Theme` is returned by `Theme.default()`.

It is possible to pass the `debug_info` dict that the `step()` method returns back to the `render()` method, this will display some useful debugging information on screen.

#### Action Space
A tuple of 2, 1-hot encoded actions.
```python
Tuple[Discrete, Discrete]
```
The first action is the acceleration in the X-dimensions, the second action is the acceleration in the Y-dimensions.
The number of available actions for each dimension is `(2 * len(acceleration_levels)) + 1`, since `len(acceleration_levels)` defaults to 3, the default number of available action per dimension is 7.
The middle action always indicates a no-op (No acceleration, existing velocity will continue to move the submarine), actions with indices lower than the middle move towards the top-left of the world, actions with indices greater than the middle move towards the bottom-right of the world.
When an action would cause a collision the sub's velocity is set to 0 in both dimensions and no action occurs.

#### Observation Space
```python
Box
```
The observation in this environment is a 2 x N matrix, with N equal to `len(treasure_values) + 1`.
The first column in the matrix (`obs[:, 0]`) contains the submarine's current velocity.
The next N columns contain the submarine's position, relative to each of the treasures.

#### Reward Space
```python
Box
```
The reward for this environment is a 2-element vector.
The first element (`reward[0]`) contains the treasure reward. This will always be 0, unless the submarine is on a treasure square.
The second element (`reward[1]`) contains the time reward, this reward is always -1, unless the submarine is on a treasures square.

### [`FuelWrapper`](deep_sea_treasure/fuel_wrapper.py)
```python
from deep_sea_treasure import FuelWrapper
```
The `FuelWrapper` is a wrapper for the DeepSeaTreasureV0 environment that adds a third objective, fuel consumption.
Fuel consumption is similar to time, in the sense that the reward for this will be negative at each timestep.
It differs from time in the sense that the consumed fuel depends on the action taken.
Accelerating 1 in either dimension will usually cost 1 fuel, but accelerating 3 usually costs 3 fuel.
The addition of this mechanic makes high accelerations followed by coasting an attractive strategy.
Creation of this wrapper is handled by the `new()` static method.
This is the recommended way of creating wrappers, since it provides meaningful arguments for changing the various settings for the environment.
Beyond this, there is also a constructor that accepts a [`gym.Env`](https://gym.openai.com/docs/#environments) and a dictionary of settings,
this was mainly done to retain compatibility with RL frameworks like RLLib.
Internally, the `new()` method simply fills a dictionary, and delegates to the constructor.
The constructor uses JSON schemas along with a couple of extra `assert`s to verify the correctness of a given configuration.
The exact schema can be found in [fuel_wrapper.schema.json](deep_sea_treasure/schema/fuel_wrapper.schema.json).

| Option      | Type        | Meaning |
| ----------- | ----------- | ------- |
| `fuel_cost` | `List[int]` | The cost of each `acceleration_level` in fuel units. The no-op action is always assumed to consume no fuel. The length of this list should always match the length of the `acceleration_levels` list in the core environment.

#### Action Space
_Same as DeepSeaTreasureV0._

#### Observation Space
_Same as DeepSeaTreasureV0._

#### Reward Space
```python
Box
```
The reward for this environment is a 3-element vector. The first 2 elements are identical to those in the `DeepSeaTreasureV0` environment.
The first element (`reward[0]`) contains the treasure reward. This will always be 0, unless the submarine is on a treasure square.
The second element (`reward[1]`) contains the time reward, this reward is always -1, unless the submarine is on a treasures square.
The third element (`reward[2]`) contains the fuel reward, this reward reflects the fuel consumed by the last action the agent took.

### [`VamplewWrapper`](deep_sea_treasure/vamplew_wrapper.py)
```python
from deep_sea_treasure import VamplewWrapper
```
The `VamplewWrapper` is a wrapper intended to undo the modifications we made to the core [`DeepSeaTreasureV0`](deep_sea_treasure/deep_sea_treasure_v0.py) environment.
It wraps both the action and observation space, to make sure the environment matches the original setup by [Vamplew et al.](https://doi.org/10.1007/s10994-010-5232-5) exactly.
This means that the `VamplewWrapper` has a different action and observation space from the original `DeepSeaTreasureV0` environment.
The `VamplewWrapper` can wrap the `FuelWrapper`, but not the other way around, due to action-space incompatibility.
Creation of this wrapper is handled by the `new()` static method.
This is the recommended way of creating wrappers, since it provides meaningful arguments for changing the various settings for the environment.
Beyond this, there is also a constructor that accepts a [`gym.Env`](https://gym.openai.com/docs/#environments) and a dictionary of settings,
this was mainly done to retain compatibility with RL frameworks like RLLib.
Internally, the `new()` method simply fills a dictionary, and delegates to the constructor.
The constructor uses JSON schemas along with a couple of extra `assert`s to verify the correctness of a given configuration.
The exact schema can be found in [vamplew_wrapper.schema.json](deep_sea_treasure/schema/vamplew_wrapper.schema.json).


| Option        | Type   | Meaning |
| ------------- | ------ | ------- |
| `enable_idle` | `bool` | When true, this option enables a 5th action in this environment, idle. This allows the submarine to sit still and do nothing.

#### Action Space
The action space for the Vamplew wrapper consists of a single 1-hot encoded action.
There are 4 or 5 possible actions to take, depending on how the wrapper was configured:
- Up
- Right
- Down
- Left
- (Idle)

Actions are specified in this order.
Each action will cancel out all velocity from the previous action, and make the velocity in the desired direction 1.


#### Observation Space
```python
Box
```
The observation in this environment is a 2-element vector, containing the submarine's current row and column.

#### Reward Space
_Same as DeepSeaTreasureV0._
