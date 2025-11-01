import ale_py
import gymnasium as gym
import numpy as np

from basic import *
from player import *

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('Breakout-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale")

model = model_new()
eval = BreakoutProgrammable(model=model,debug=False) 

scores = []
stepss = []
livess = []
states = []

steps = 0
score = 0
lives = None

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

max_steps = 18000 # 18000 # according to Igor Pivoarov! (but games are truncated at 108000) 
max_games = 100
game = 0
reward = 0
action = None

# Reset the environment to generate the first observation
#observation, info = env.reset(seed=42)
observation, info = env.reset()
while (game < max_games):
    # this is where you would insert your policy
    if action is None:
        action = 2 # env.action_space.sample() # TODO why setting to 0 or 1 crashes on start?
    else:
        action = eval.process_state(observation, reward, action) # pass previous action in 

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if lives == None: # Breakout-specific!!!
        lives = info['lives'] # initial amount of lives
    reward -= (lives - info['lives']) # decrement reward by "lost life", if the life is lost, according to Igor Pivovarov
    lives = info['lives']

    if reward != 0:
        score += reward
        if reward < 0:
            print(reward,info['lives'],score,scores)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated or steps == max_steps:
        observation, info = env.reset()
        scores.append(score)
        stepss.append(steps)
        livess.append(lives)
        score = 0
        steps = 0 
        lives = None
        # TODO action = 1 !?
        print('terminated' if terminated else 'truncated' if truncated else f'{max_steps} steps limit')
        print('scores =', scores, round(np.mean(scores),1))
        print('steps =', stepss, round(np.mean(stepss),1))
        print('lives =', livess, round(np.mean(livess),1))
        if not model is None:
            states.append(len(model['states']))
            print('states =', states)
            model_write_file(f'programmatic{game}',model)
        print('==============')
        game += 1

env.close()
