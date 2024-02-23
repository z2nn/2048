import torch
import zmq
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import torch as th
import os

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5554")

LOG_DIR = "logs/2048"

os.makedirs(LOG_DIR, exist_ok=True)

class CustomCNN2(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # 定义你的卷积层
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def save_model(model, path="trained_models/2048_model.zip"):
    # 确保保存模型的目录存在，如果不存在，则创建
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 保存模型
    model.save(path)
    print(f"Model saved to {path}")


def send_message(action):
    # Convert action to int before sending
    action = int(action)
    socket.send_json({"action": action})
    message = socket.recv_json()
    return message


class ChaseEnv(gym.Env):
    def __init__(self):
        super(ChaseEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 4, 4), dtype=float)

    def step(self, action):
        response = send_message(action)
        state = np.array(response["game_state"])
        state_p = np.where(state <= 0, 1, state)
        state = np.log2(state_p) / np.log2(2 ** 16)
        reward = response["reward"]
        done = response["ifEnd"] == 1
        info = {}
        return state.reshape(1, 4, 4), reward, done, 0, info

    def reset(self, seed=None, options=None):
        response = send_message(5)
        state = np.array(response["game_state"])
        state_p = np.where(state <= 0, 1, state)
        state = np.log2(state_p) / np.log2(2 ** 16)
        info = {}
        return state.reshape(1, 4, 4), info


policy_kwargs = dict(
    features_extractor_class=CustomCNN2,
    features_extractor_kwargs=dict(features_dim=512),
)


def train_model(env):
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=LOG_DIR)  # 使用多层感知机作为策略网络
    model.learn(total_timesteps=10000000)
    return model


# Test the trained model
def test_model(model, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    env = ChaseEnv()
    model = train_model(env)
    # model = DQN.load("trained_models/2048_model.zip")
    save_model(model)
    # test_model(model, env)