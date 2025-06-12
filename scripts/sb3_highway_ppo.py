import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import sys
sys.path.append("D:/AUTO_Drive/HighwayEnv-master-0428/HighwayEnv-master-0428/")
import highway_env  # noqa: F401
highway_env._register_highway_envs()



# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = False
    #train = True
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env("complex_v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size * 12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )
        # Train the agent
        model.learn(total_timesteps=int(100000))
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")
    env = gym.make("complex_v0", render_mode="rgb_array")
    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
