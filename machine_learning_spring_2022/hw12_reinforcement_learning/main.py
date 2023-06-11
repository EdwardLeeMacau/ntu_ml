"""
## Reference

Below are some useful tips for you to get high score.

- [DRL Lecture 1: Policy Gradient (Review)](https://youtu.be/z95ZYgPgXOY)
- [ML Lecture 23-3: Reinforcement Learning (including Q-learning) start at 30:00](https://youtu.be/2-JNBzCq77c?t=1800)
- [Lecture 7: Policy Gradient, David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
"""

import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import PolicyGradientAgent, PolicyGradientNetwork
from torch.distributions import Categorical
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from utils import same_seeds

"""
# Warning ! Do not revise random seed !!!
# Your submission on JudgeBoi will not reproduce your result !!!
Make your HW result to be reproducible.
"""

seed = 543
def fix_env(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)

env = gym.make('LunarLander-v2')
env.reset()

fix_env(env, seed) # fix the environment Do not revise this !!!
same_seeds(seed)

network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

def train():
    agent.network.train()  # Switch network into training mode
    EPISODE_PER_BATCH = 5  # update the agent every 5 episode
    NUM_BATCH = 500        # totally update the agent for 400 time

    avg_total_rewards, avg_final_rewards = [], []

    pbar = tqdm(range(NUM_BATCH), ncols=0)
    for batch in pbar:
        log_probs, rewards = [], []
        total_rewards, final_rewards = [], []

        # collect trajectory
        for episode in range(EPISODE_PER_BATCH):
            state = env.reset()
            total_reward, total_step = 0, 0
            seq_rewards = []

            while True:
                action, log_prob = agent.sample(state) # at, log(at|st)
                next_state, reward, done, _ = env.step(action)

                log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
                # seq_rewards.append(reward)
                state = next_state
                total_reward += reward
                total_step += 1
                rewards.append(reward) # change here

                # ! IMPORTANT !
                # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
                #                                                         rewards :     r1, r2 ,r3 ......
                # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
                #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
                # boss : implement Actor-Critic

                if done:
                    final_rewards.append(reward)
                    total_rewards.append(total_reward)

                    break

        print(f"rewards looks like ", np.shape(rewards))
        #print(f"log_probs looks like ", np.shape(log_probs))
        # record training process
        avg_total_reward = sum(total_rewards) / len(total_rewards)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        avg_total_rewards.append(avg_total_reward)
        avg_final_rewards.append(avg_final_reward)
        pbar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

        # update agent
        # rewards = np.concatenate(rewards, axis=0)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
        print("logs prob looks like ", torch.stack(log_probs).size())
        print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

def test():
    fix_env(env, seed)
    agent.network.eval()  # set the network into evaluation mode
    NUM_OF_TEST = 5 # Do not revise this !!!
    test_total_reward = []
    action_list = []
    for i in range(NUM_OF_TEST):
        actions = []
        state = env.reset()

        img = plt.imshow(env.render(mode='rgb_array'))

        total_reward = 0

        done = False
        while not done:
            action, _ = agent.sample(state)
            actions.append(action)
            state, reward, done, _ = env.step(action)

            total_reward += reward

            img.set_data(env.render(mode='rgb_array'))
            # display.display(plt.gcf())
            # display.clear_output(wait=True)

        print(total_reward)
        test_total_reward.append(total_reward)

        action_list.append(actions) # save the result of testing

    print(np.mean(test_total_reward))

    """Action list"""

    print("Action list looks like ", action_list)
    print("Action list's shape looks like ", np.shape(action_list))

    """Analysis of actions taken by agent"""

    # Modify data structure from dict to counter
    distribution = {}
    for actions in action_list:
        for action in actions:
            if action not in distribution.keys():
                distribution[action] = 1
            else:
                distribution[action] += 1
    print(distribution)

    PATH = "Action_List.npy" # Can be modified into the name or path you want
    np.save(PATH, np.array(action_list))

    # Evaluate model using fixed random seed
    action_list = np.load(PATH, allow_pickle=True) # The action list you upload
    seed = 543 # Do not revise this
    fix_env(env, seed)

    agent.network.eval()  # set network to evaluation mode

    test_total_reward = []
    if len(action_list) != 5:
        print("Wrong format of file !!!")
        exit(0)

    for actions in action_list:
        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))

        total_reward = 0

        done = False

        for action in actions:
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        print(f"Your reward is : %.2f"%total_reward)
        test_total_reward.append(total_reward)

    """# Your score"""

    print(f"Your final reward is : %.2f"%np.mean(test_total_reward))

if __name__ == '__main__':
    train()
