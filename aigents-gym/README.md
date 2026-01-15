# State-based History-aware Artificiall Reinforcement Intelligence Kernel (Sharik) - Files, Classes, TODO and Diary 

## Files

- [breakout_hack_simple.py](./breakout_hack_simple.py) - "cheating" Breakout player, knows rules of the game and properties of the game field, wins 860 (out of top 864 points) always if not limited by number of steps per game
- [breakout_hack_simple1.py](./breakout_hack_simple1.py) - same as above logic wrapped in BreakoutHacky

## Classes

- player.BreakoutHacky - "cheating" Breakout player, knows rules of the game and properties of the game field
  - if action = env.action_space.sample() at game (re)start: wins 831/860 (out of top 864 points, varies even with fixed random seed) if not limited by number of steps per game (108000 steps max) or 616/725/732/856 (varies even with fixed random seed) if limited by 18000 steps
  - if action = 1 at game (re)start: wins 860 (out of top 864 points) always (regardless of random seed) if not limited by number of steps per game (108000 steps max) or 732 (regardless of random seed) if limited by 18000 steps
- ...


## TODO

- play with different random seeds
  - find "the right (stable) seed" to play with HP - PROGRESS
- play with HP for the "best model" to make it learning more stable and predictable
  - SC = 3 (TODO commit)
  - remove reward/punishment from state?
  - represent action as 5 "hots" so
     - similarity is computed more accurately
     - we can correlate actions with moves (add Xracket derivative?)  
  - utility vs count vs utility * count - for the "best model" to improve it
  - replace Xr+Cb with Dx  
  - denominate U by number of states (energy spent)?
- make sure the performance is seed-agnostic
  - run up to 3-5K with three different seeds for SS=0.9 and SS=0.95, to collect the difference!? 
- remove action = 1 # HACK, replace with action = env.action_space.sample() (inter-play 1-FIRE hardcoding)
- model on racket_x - ball_x, racket_speed, ball_speed - PROGRESS
  - python ./aigents-gym/breakout_eval2.py -cs=2 -ss=0.9 -tu=0 -s=41 -o=202501112_relx_s41

- find_usefulNov32025 - random ties!

- draft paper of ICML https://icml.cc/
  - ...

 
- TODO see how "model compression" can affect runtime performance - model_pack.py

- TODO see how "best models" perform in real time
  - python ./aigents-gym/breakout_eval1.py -i=model_Nov32025_PN_CS2_SS099_251117w -cs=2 -ss=0.99 -tu=0 

- fix for all models:
  -  if self.state_similarity_threshold < 1.0:

- add energy-efficiency (based on step count) to improve it
- start removing hardcoding the "computer vision" and see how it works
- try to feedback by fact of reward, not by amount of reward

- TODO test old code, and unitfy with lateset code!!!

- best_action CS=2 ?

- negative feedback on steps consumed

- provide negative feedback not by failed game, but evaluate overall based on number of steps? 

- explore correlation and ganger causation across state variables and cluster them on this matter!?
- restart episodes on uncertainty, to learn only certain episodes!!!???
- dynamically learn T>1 episodes based on probability!?

- debug find_similar_action

- across transitions - compute grand utility per action weighted by evidence counts
- include action with no transitions to the list with 0 utility 

- small negative feedbacks for each step!?
- small internal feedback for finding novel situations!!!???



## Diary

- ... 

## References

- https://arxiv.org/abs/2509.07009
- https://arxiv.org/abs/2512.10985
- https://arxiv.org/abs/2511.22130
- 
