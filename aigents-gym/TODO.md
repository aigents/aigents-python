# State-based History-aware Artificial Reinforcement Intelligence Kernel (Sharik) 1.2 - TODO

## TODO NEXT
- goals
  - per/step evaluation
  - "learning stabilty"
  - "representation learning" https://www.nature.com/articles/s41592-025-02729-9 Interpretable representation learning for 3D multi-piece intracellular structures using point clouds
  - "energy-awareness"
  - "model compression"
  - "multiple environments"
  - "cross-environmental generalization"

- learning stability
  - run 1M steps/frames with different seeds, find average score
  - run this with different seeds, find "sweet seed"

- figure out step/frame difference in https://arxiv.org/pdf/1911.08265
- RE-play with all the HP for the "best model" to make it learning more stable and predictable (based on step/frame counts)
  - limit training/learning by number of steps (not games) in Millions, count won games!!!???
  - redo all other HPs
  - replace Xr+Cb with Dx  
  - denominate U by number of states (energy spent)?
- model on racket_x - ball_x, racket_speed, ball_speed (HOLD)
  - python ./aigents-gym/breakout_eval2.py -cs=2 -ss=0.9 -tu=0 -s=41 -o=202501112_relx_s41
- remove action = 1 # HACK, replace with action = env.action_space.sample() (inter-play 1-FIRE hardcoding)
- run 3-4 different seeds for 10K games (with the best HP!) - Round 4!!!

- find_usefulNov32025 - random ties!

- TODO see how "model compression" can affect runtime performance - model_pack.py

- TODO see how "best models" perform in real time
  - python ./aigents-gym/breakout_eval1.py -i=model_Nov32025_PN_CS2_SS099_251117w -cs=2 -ss=0.99 -tu=0 

- fix for all models:
  -  if self.state_similarity_threshold < 1.0:

- add energy-efficiency (based on step count) to improve it
- start removing hardcoding the "computer vision" and see how it works
- try to feedback by fact of reward, not by amount of reward

- TODO test old code, and unitfy with lateset code!!!

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


## References

- https://link.springer.com/chapter/10.1007/978-3-030-93758-4_12 Experiential learning Kolonin
- https://arxiv.org/abs/2509.07009 Psyche Kolonin
- https://arxiv.org/abs/2512.10985 Marti
- https://arxiv.org/abs/2511.22130 Experiential learning
- https://dl.acm.org/doi/10.1109/TPAMI.2013.50 Yoshua Bengio, Representation Learning: A Review and New Perspectives
- https://dl.acm.org/doi/10.5555/3454287.3455074 Yoshua Bengio, Unsupervised state representation learning in atari
- https://www.mdpi.com/2075-1702/13/12/1140 Reinforcement Learning for Industrial Automation: A Comprehensive Review of Adaptive Control and Decision-Making in Smart Factories
 - https://ieeexplore.ieee.org/document/9619637 Opportunities for Reinforcement Learning in Industrial Automation 
 - https://arxiv.org/abs/2502.09417v1 A Survey of Reinforcement Learning for Optimization in Automation
 - https://dl.acm.org/doi/abs/10.5555/2566972.2566979 Marc G. Bellemare, The arcade learning environment: an evaluation platform for general agents
 - https://arxiv.org/pdf/1312.5602 Mnih 2013, Playing Atari with Deep Reinforcement Learning
 - https://arxiv.org/abs/1908.04683 - Implicit Quantile Networks (IQN), Rainbow-IQN, Toromanoff 2019, Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the playing field
- https://arxiv.org/abs/2002.06038 - NGU: 576-864
- https://arxiv.org/pdf/2003.13350 - Agent57: 790 (multi-agent), Badia, 2020, Agent57: Outperforming the Atari Human Benchmark,
- https://arxiv.org/abs/1911.08265 MuZero: 864, planning, hidden state, Schrittwieser 2020, Mastering Atari, Go, chess and shogi by planning with a learned model

