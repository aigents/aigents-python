import ale_py
import gymnasium as gym
import numpy as np
import datetime as dt

from basic import *
from player import *

# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human") # works

# https://gymnasium.farama.org/v0.28.0/environments/atari/breakout/
#env = gym.make('Breakout-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human') # works
#env = gym.make('BreakoutNoFrameskip-v4', render_mode='human', obs_type="grayscale") 
env = gym.make('BreakoutNoFrameskip-v4', obs_type="grayscale")

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

#model = model_new()
#eval = BreakoutProgrammable(model=model,debug=False)
model = None
eval = BreakoutModelDriven(list(range(env.action_space.n)),model=model_read_file("./models/breakout/programmatic99"),debug=False) 

scores = []
stepss = []
livess = []
states = []
lapses = []

steps = 0
score = 0
lives = None

max_steps = 18000 # 18000 # according to Igor Pivoarov! (but games are truncated at 108000) 
max_games = 10
game = 0
reward = 0
action = None

# Reset the environment to generate the first observation
#observation, info = env.reset(seed=42)
t0 = dt.datetime.now()
observation, info = env.reset()
while (game < max_games):
    # this is where you would insert your policy
    if action is None:
        action = 2 # env.action_space.sample() # TODO why setting to 0 or 1 crashes on start?
    else:
        action = eval.process_state(observation, reward, action) # pass previous observation, reward (may be negative) and past action (remembered) in 

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if lives == None: # Breakout-specific!!!
        lives = info['lives'] # initial amount of lives
    reward -= (lives - info['lives']) # decrement reward by "lost life" if the life is lost (to pass it to the next process_state)!
    lives = info['lives']

    if reward > 0: #TODO don't subtract lives from rewards!
        score += reward
    elif reward < 0:
        print(reward,info['lives'],score,scores)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated or steps == max_steps:
        t1 = dt.datetime.now()
        lapse = t1 - t0
        t0 = t1
        observation, info = env.reset()
        scores.append(score)
        stepss.append(steps)
        livess.append(lives)
        lapses.append(round(lapse.total_seconds()))
        score = 0
        steps = 0 
        lives = None
        # TODO action = 1 !?
        print(f"cause=\"{'terminated' if terminated else 'truncated' if truncated else f'{max_steps}_steps_limit'}\"; " +
              f"score={round(np.mean(scores),1)}; steps={round(np.mean(stepss),1)}; lives={round(np.mean(livess),1)}; lapse=\"{str(lapse)}\"")
        print('scores =', scores)
        print('stepss =', stepss)
        print('livess =', livess)
        print('lapses =', lapses)
        if not model is None:
            states.append(len(model['states']))
            print('states =', states)
            model_write_file(f'programmatic{game}',model)
        print('==============')
        game += 1

env.close()
