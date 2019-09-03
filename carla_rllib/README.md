# CARLA Reinforcement Learning Library (work under progress)

This library supports reinforcement learning in the CARLA Simulator.

## Installation

* Clone this repository to your local machine:

```konsole
$ git clone https://ids-git.fzi.de/svmuelle/carla_rllib.git
```
**Optional**: add `-b carla_env_0.9.5` in case you are using carla version 0.9.5 **(this version won't be updated)**

* Install the required python packages (into your virtual environment):

```console
$ cd carla_rllib
$ pip install -r requirements.txt
```

* Add carla_rllib to your PYTHONPATH:

```console
$ export CARLA_RLLIB=your/path/to/carla_rllib
$ export PYTHONPATH=$CARLA_RLLIB
```

## Structure

Carla environment:
* supports single- and multi-agent reinforcement learning
* configured by `config.json`
* allows frame skipping
* supports [stable baselines](https://stable-baselines.readthedocs.io/en/master/) (NOTE: single-agent only; action type changes to list)

Carla wrapper:
* supports continuous (steer, throttle) and discrete (transformation/teleportation) action control
* uses gpu-based sensors (images) and cpu-based sensors (collision, lane invasion, ...)
* agents' state is summarized in BaseState class 

Examples:
* run `carla_env_test.py` to test single- or multi-agent environments without policies
* run `carla_env_baselines_test` to test or train with stable baselines

## How to use?

In order to fit your learning goals a few things may need to be adjusted. Here are the essential parts to have a look at.

1. Configuration: `config.json`
    * Configures the environment setup such as single- or multi-agent, frame skipping, map and more
2. Environment: `carla_env.py`
    * Define the observations by extracting the desired information from the agent's state in `_get_obs()`
    * Calculate Reward based on the state of an agent in `_calculate_reward()`
    * Adjust reset information in `_get_reset()`
    * Adjust the action/observation space to support stable baseline learning if necessary in `__init__()`
3. Carla Wrapper: `carla_wrapper.py`
    * Choose spawn point and sensors in `_start()` (BaseWrapper)
    * Adjust the state update based on sensors and non-sensor data in `update_state()` (BaseWrapper)
    * Adjust the controls if necessary in `step()` (ContinuousWrapper/DiscreteWrapper)
    * Adjust the reset if necessary in `reset()` (ContinuousWrapper/DiscreteWrapper)
4. States: `states.py`
    * define a custom state that fits your learning goals
    * always adjust `update_state()` of the BaseWrapper if you changed the state

Run environments like a gym environment and always use **try-finally block** to properly destroy the carla actors:

```python
import ...

env = make env

try:
    
    training loop
    
finally:
    env.close()

```