## Files

- [LICENSE](./LICENSE) - MIT license file
- [breakout_hack_simple.py](./breakout_hack_simple.py) - "cheating" Breakout player, knows rules of the game and properties of the game field, wins 860 (out of top 864 points) always if not limited by number of steps per game (for the reference)
- [breakout_hack_simple1.py](./breakout_hack_simple1.py) - same as above "cheating" logic wrapped in BreakoutHacky class (for the reference)
- [breakout_eval2.py](./breakout_eval2.py) - main evaluation script, see script code for command line options
- [player.py](./player.py) - main script with learning and acting functinal in repective classes decribed below    
- [moel_pack.py](./model_pack.py) - utility to analyse and restructure model file, see script code for command line options
- [basic.py](./basic.py) - basic utility functions and operation with model

## Classes

- player.BreakoutHacky - "cheating" Breakout player, knows rules of the game and properties of the game field
  - if action = env.action_space.sample() at game (re)start: wins 831/860 (out of top 864 points, varies even with fixed random seed) if not limited by number of steps per game (108000 steps max) or 616/725/732/856 (varies even with fixed random seed) if limited by 18000 steps
  - if action = 1 at game (re)start: wins 860 (out of top 864 points) always (regardless of random seed) if not limited by number of steps per game (108000 steps max) or 732 (regardless of random seed) if limited by 18000 steps
- player.BreakoutXXProgrammable - "Automated" player which transforms input observations to X coordinates of pixel clouds correspondingg to horizontal positions of the ball and the racket
- player.BreakoutProgrammable - "Automated" player with extra ability to learn the model states
- player.BreakoutModelDriven - "Model-based" player - newer version, provides worse performance so far 
- player.BreakoutModelDrivenNov32025 - "Model-based" player - old version, provides the best performance so far 
