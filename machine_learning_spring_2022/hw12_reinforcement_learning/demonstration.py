from matplotlib import pyplot as plt
from IPython import display
import gym

seed = 543
def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)

# fix the environment Do not revise this !!!
env = gym.make('LunarLander-v2')
fix(env, seed)

print(env.observation_space)
print(env.action_space)

"""`Discrete(4)` implies that there are four kinds of actions can be taken by agent.
- 0 implies the agent will not take any actions
- 2 implies the agent will accelerate downward
- 1, 3 implies the agent will accelerate left and right

Next, we will try to make the agent interact with the environment.
Before taking any actions, we recommend to call `reset()` function to reset the environment. Also, this function will return the initial state of the environment.
"""

initial_state = env.reset()
print(initial_state)

"""Then, we try to get a random action from the agent's action space."""

random_action = env.action_space.sample()
print(random_action)

"""More, we can utilize `step()` to make agent act according to the randomly-selected `random_action`.
The `step()` function will return four values:
- observation / state
- reward
- done (True/ False)
- Other information
"""

observation, reward, done, info = env.step(random_action)

print(done)

"""### Reward

> Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
"""

print(reward)

"""### Random Agent
In the end, before we start training, we can see whether a random agent can successfully land the moon or not.
"""

env.reset()

img = plt.imshow(env.render(mode='rgb_array'))

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
