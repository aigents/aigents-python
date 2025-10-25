import numpy as np
from queue import Queue, Full, Empty

memory_size = 10
background_refresh_rate = 10

observations = Queue(maxsize=memory_size) 
epoch = 0


def debug_array2str(a,t):
    return ''.join(['.' if i < t else '█' for i in a])

def print_debug(array2d):
    for a in array2d:
        #print(a)
        print(debug_array2str(a,1))
    #print(len(array2d),len(array2d[0]))

debug = True

def process_state(observation, reward):
    global epoch
    global debug

    #accumulate observations 
    if observations.qsize() == memory_size:
        observations.get()
    observations.put((observation, reward))
    epoch += 1

    if epoch % background_refresh_rate == 0:
        observation_maps = [a[0] for a in list(observations.queue)] # grayscale!
        average_array = np.mean(observation_maps, axis=0)
        diff = np.maximum(np.subtract(observation,average_array),0)
        if debug:
            #print_debug(observation) # OK
            print_debug(diff)
            print('===')
            debug = False

    return 3 if debug_count % 5 != 0 else 1





import ale_py
import gymnasium as gym

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('Breakout-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human') # works
env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 


# For discrete action spaces (like Atari games)
if hasattr(env.action_space, 'n'):
    print(f"Total actions: {env.action_space.n}")
    print("All possible actions:", list(range(env.action_space.n)))

# Get action meanings
if hasattr(env, 'get_action_meanings'):
    action_meanings = env.get_action_meanings()
    print("Action meanings:", action_meanings)
    # Create a mapping of action numbers to their meanings
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")

debug_count = 0

action = None

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(10000):
    # this is where you would insert your policy
    if action is None:
        action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    if reward > 0:
        print(reward)

    debug_count += 1
    if debug_count % 100 == 0:
        #print(observation)
        pass

    action = process_state(observation, reward)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
