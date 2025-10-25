import sys
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
rocket_row = None
diff_vert = None
average_array = None 

def process_state(observation, reward):
    """
    Value Meaning
    0 NOOP
    1 FIRE
    2 RIGHT
    3 LEFT
    """
    global epoch
    global debug
    global average_array

    # accumulate observations 
    if observations.qsize() == memory_size:
        observations.get()
    observations.put((observation, reward))
    epoch += 1

    # update background
    if epoch % background_refresh_rate == 0: 
        observation_maps = [a[0] for a in list(observations.queue)] # grayscale!
        average_array = np.mean(observation_maps, axis=0)
    
    if average_array is None:
        act = 0
    else:
        diff = np.maximum(np.subtract(observation,average_array),0)
        global rocket_row
        global diff_vert
        if rocket_row is None:
            max = 0
            diff_vert = [int(np.sum(d)) for d in diff] 
            for row in range(len(diff_vert)):
                if diff_vert[row] > max:
                    max = diff_vert[row]
                    rocket_row = row
            print(rocket_row)
        diff_ball = diff[0:rocket_row]
        ball_col = np.argmax(np.convolve(np.mean(diff_ball, axis=0), [1,1,1], mode='same'))
        rocket_col = np.argmax(np.convolve(diff[rocket_row], [1,1,1], mode='same'))
        
        if rocket_col < ball_col:
            act = 2 # RIGHT
        elif rocket_col > ball_col:
            act = 3 # LEFT
        else:
            act =1

        if debug and rocket_col == -1:
            #print_debug(observation) # OK - binary map raw
            print_debug(diff) # OK - binary map of ball and rocket
            print(diff_vert)
            print('rocket_row',rocket_row)
            print(diff[rocket_row])
            print('===')
            try:
                input("Press enter to continue")
            except SyntaxError:
                pass

        if debug: 
            print(ball_col,rocket_col,act)

    return act





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
