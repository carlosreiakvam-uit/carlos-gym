import random

import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

nothing, left, up, right = 0, 1, 2, 3
xPos, yPos, xVel, yVel, angle, angle_vel, leg1_ground, leg2_ground = 0, 1, 2, 3, 4, 5, 6, 7


def policy():
    a = env.observation_space.sample()
    if random.random() < 0.5:
        if random.random() < 0.5:
            return right
        else:
            return left
    else:
        return up


reward_sum = 0
for _ in range(1200):
    action = policy()

    observation, reward, terminated, truncated, info = env.step(action)

    reward_sum += reward
    if terminated or truncated:
        print(reward_sum)
        reward_sum = 0
        observation, info = env.reset()
env.close()
