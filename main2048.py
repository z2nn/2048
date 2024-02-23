import zmq
import json
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
import os

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5554")


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
        self.action_space = spaces.Discrete(4) # 上、下、左、右
        self.observation_space = spaces.Box(low=0, high=2**12, shape=(4, 4), dtype=int)

    def step(self, action):
        response = send_message(action)
        state = np.array(response["game_state"])
        reward = response["reward"]
        done = response["ifEnd"] == 1
        info = {}
        return state, reward, done, info

    def reset(self):
        response = send_message(5)
        state = np.array(response["game_state"])
        return state

def train_model(env):
    model = QRDQN("MlpPolicy", env, verbose=2)  # 使用多层感知机作为策略网络
    model.learn(total_timesteps=1000000)  # 训练10000步
    return model

# Test the trained model
def test_model(model, env, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(state)  # 使用训练好的模型预测动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    env = ChaseEnv()
    model = train_model(env)
    save_model(model)
    test_model(model, env)