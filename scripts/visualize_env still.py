import gymnasium as gym
import matplotlib.pyplot as plt
import highway_env
import highway_env
import sys
print(highway_env.__file__)

# 替换成你注册的自定义环境ID
env = gym.make("example-v0", render_mode="rgb_array")
obs, _ = env.reset()

# 渲染当前状态
img = env.render()

# 用matplotlib显示图像
plt.imshow(img)
plt.axis("off")
plt.title("Custom Complex Env")
plt.show()
