# State-based History-aware Artificiall Reinforcement Intelligence Kernel (Sharik) - TODO and Diary 

## TODO

- TODO play with different rendom seeds
  - make random seed working for learning
  - find "the right (stable) seed" to play with HP
  - make sure the perfromance is seed-agnostic
  - remove action = 1 # HACK

- TODO draft paper of ICML https://icml.cc/
  - ...
 

- TODO play with HP for the "best model" to make it learning more stable and predictable
  - utility * count ?
  - SC

#TODO see how "model compression" can affect performance - model_pack.py

#TODO see how "best models" perform in real time
## python ./aigents-gym/breakout_eval1.py -i=model_Nov32025_PN_CS2_SS099_251117w -cs=2 -ss=0.99 -tu=0 

#TODO fix for all models:
#                if self.state_similarity_threshold < 1.0:

#TODO play with utility vs count for the "best model" to improve it
#TODO add energy-efficiency (based on step count) to improve it
#TODO remove inter-play 1-FIRE hardcoding 
#TODO start removing hardcoding the "computer vision" and see how it works
#TODO try to feedback by fact of reward, not by amount of reward



#TODO test old code, and unitfy with lateset code!!!

#TODO best_action CS=2

#TODO negative feedback on steps consumed

#TODO provide negative feedback not by failed game, but evaluate overall basded on number of steps? 

#Test new code on Ut = None
#Test new code on U vs U*C

#TODO explore correlation and ganger causation across state variables and cluster them on this matter!?
#TODO restart episodes on uncertainty, to learn only certain episodes!!!???
#TODO dynamically learn T>1 episodes based on probability!?


#TODO pass parameters at cmd line
#TODO debug find_similar_action
#TODO model on racket_x - ball_x, racket_speed, ball_speed
#TODO T=1 => T=2

# TODO!?
# across transitions - compute grand utility per action weighted by evidence counts
# include action with no transitions to the list with 0 utility 
# new ST=0.99999
# new SS=0.9

# TODO small negative feedbacks for each step!?
# TODO small internal feedback for finding novel situations!!!???



## Diary

### ... 

## References

- https://arxiv.org/abs/2509.07009
- https://arxiv.org/abs/2512.10985
- https://arxiv.org/abs/2511.22130
- 
