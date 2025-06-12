# 🚗 Complex_road_env

A modular autonomous driving simulation environment based on [HighwayEnv](https://github.com/eleurent/highway-env), designed for complex traffic scenarios.

## 🧩 Features

- **Modular Roads**:  
  - `straight`, `merge`, `double_u_turn`, `intersection`, `parking`

- **Smooth Connectivity**:  
  - Continuous road structure using cumulative coordinates

- **Reward Design**:  
  - Driving stage: lane-keeping, right-lane preference  
  - Parking stage: position & heading accuracy

- **RL-Ready**:  
  - Compatible with Stable-Baselines3 for training (DDPG, PPO, SAC)

## 🚀 Quick Start

```python
import gymnasium as gym
from stable_baselines3 import DDPG

env = gym.make("complex-v0")
model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ddpg_complex")
