import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt

# 创建环境并配置为灰度观测
env = gym.make(
    "highway-fast-v0",
    config={
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),  # 图像尺寸
            "stack_size": 4,  # 堆叠帧数
            "weights": [0.2989, 0.5870, 0.1140],  # RGB转灰度权重
            "scaling": 1.75,  # 缩放比例
        },
    },
)

# 重置环境并获取初始观测
obs, info = env.reset()

# 执行几步动作，确保堆叠帧中有有效数据
for _ in range(20):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())

# 将归一化的数据转换到 [0, 255]
obs_uint8 = (obs[0] * 255).astype('uint8')

# 显示灰度图像
plt.imshow(obs_uint8, cmap='gray')
plt.title("Grayscale Observation")
plt.axis('off')
plt.show()

# 关闭环境
env.close()