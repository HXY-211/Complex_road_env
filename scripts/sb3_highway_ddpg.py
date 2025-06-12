from calendar import TUESDAY
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
import numpy as np
sys.path.append("D:/AUTO_Drive/HighwayEnv-master-0428/HighwayEnv-master-0428/")
import highway_env  # noqa: F401
highway_env._register_highway_envs()

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    # train = False
    train = False
    if train:
        n_cpu = 6
        # DDPG通常不需要多环境并行（因其为Off-Policy算法）
        env = gym.make("complex_v0")
        
        # 获取动作空间的维度（DDPG需要）
        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
            theta=0.15,
            dt=1e-2
        )

        model = DDPG(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            learning_rate=5e-4,
            buffer_size=100000,
            batch_size=64,
            action_noise=action_noise,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ddpg/",
        )
        # Train the agent
        model.learn(total_timesteps=int(10000))
        # Save the agent
        model.save("highway_ddpg/model")

    model = DDPG.load("highway_ddpg/model")
    env = gym.make("complex_v0", render_mode="rgb_array")
    for _ in range(100):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)  # DDPG输出确定性动作
            obs, reward, done, truncated, info = env.step(action)
            env.render()