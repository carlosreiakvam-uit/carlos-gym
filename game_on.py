import random
import torch
import numpy as np
from collections import deque

nothing, left, up, right = 0, 1, 2, 3
xPos, yPos, xVel, yVel, angle, angle_vel, leg1_ground, leg2_ground = 0, 1, 2, 3, 4, 5, 6, 7


def policy(env):
    a = env.observation_space.sample()
    if random.random() < 0.5:
        if random.random() < 0.5:
            return right
        else:
            return left
    else:
        return up


def run(env, policy):
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


def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            print('.', end='')
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nself.environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                              np.mean(
                                                                                                  scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores
