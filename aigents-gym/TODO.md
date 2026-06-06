# State-based History-aware Artificial Reinforcement Intelligence Kernel (Sharik) 1.2 - TODO

## TODO NEXT
- IMPROVE
  - learning stability
    - get rid of cosine_similarity with euclidean_similarity
      * re-run with new frame-based experimental setup, with different random seeds (sim_test2)
        + 1-d/max with self.similarity_dims and self.similarity_max_dist specific to S
        + exp(-d) for the same 5400000 frames as Mnih
          + python ./aigents-gym/breakout_eval2.py -cs=2/3/41 -ss=0.9 -tu=0 -s=2 -mg=50000 -mt=5400000 -sm="exp(-d)" > sim_exp/exp5400000_s2/3/41ss09.txt
        - optimize H-P!
          * with no threshold at all (ss=1.0) with wider ange of thresholds and zero-threshold
            + lm=1/2 (2)
            + ss=1.0
            + sc=1/2/3 (1-2)
            * cs=1/2/3 (2-3???)
              + 1080000 cs3
              + 5400000 cs2/cs3
              * 54000000 - ???
            * counted utility!
              + 1080000 - pu0 is better than p1
              * 5400000
            ! randomise multiple ties of useful transitions!?
    * c - chance to perform random action ("curiosity"), make dependant on "surpriziness" z_t = (s_t - s't)/max(s_t,s't), as c_t = Z * z_t
      + -cc - constant curiosity (inverse to greediness) - chance to perform random action - does not help
      ! -mc - motivated curiosity ....
    ! fix paper comparison to Mhih!?
    - plot representation "like Mhih & CEC"!?
    - cos with one-hot encoding of x and y ???
    - g - extra positive feedback for predictiviness (1 - z_t), as d_t = R * (1-z_t), where R can be considered as an element of x
      OR
      - the same for "discoverinness"?
    - "selecting actions proportionally to their value estimate, injected with Gaussian noise per action", CEC, https://arxiv.org/abs/2211.15183 
    - e - extra negative feedback for energy consumption
  - expreimental setting like in Mnih?
    - 100 epochs X epoch corresponds to 50000 minibatch weight updates or 30 minutes = 108,000 frames (averge over games??? is 168)
    - 50 epochs = 5,400,000 frames (maximum possible score of 225)
    - frameskip k = 4 
  - dimensionatilty reduction
    - pixel map C(X,Y,RGB) => grayscale c or JPEG-style!?
    - grayscale G(X,Y) => spots "spot transformation" ... !!!???
      - smooth (optional, for cross-frame similarity)?
      - cutoff triples by theshshold to binary map BW(X,Y)
      - convert binary pixel map BW(X,Y) to graph of triples (G,X,Y)
      - cluster (G,X,Y) till they can be clustered
    - as coefficient codes ... !!!???
    - to object map (Object,Property)


- Summary on review-based improvements
  - improve paper
    - ! demonstrate and explain interpretability/explainability  
    - claify the goals of the expreiments
    - clarify what is maximized, definition of U, difference between U and Q in Q-learning
    - provide pseudocode
    - report memory consumption and graph size
    - mean and standard deviation
    - explain the pixel to objects transformation and graph representation
  - new experiments
    - ! different environments
    - ! generalization from pixels
    - report based on number of steps, not games
    - standard benchmarking protocols for direct comparison with deep reinforcement learning methods
  - new math work
    - theoretical analysis or convergence
- Suggested review-based improvements
  - It would be better to show more results on different environments. The paper conducts limited evaluations, the model is tested only on one environment
  - It uses a custom computer vision algorithm for preprocessing which makes it hard for the model to generalize to other environments.
  - The models being compared (DQN, IQN) operate on raw pixels and may not provide a fair comparison.
  - The evaluation doesn't use standard benchmarking protocols (e.g., reporting mean and standard deviation across multiple seeds as is standard practice), making comparison with existing literature difficult.
  - The proposed method is explained in an unclear manner, I would request the authors to add pseudocode for their approach to simplify and strengthen the paper.
  - The paper lacks any theoretical analysis or convergence guarantees for the learning algorithm.
  - The paper claims interpretability, but does not demonstrate this through examples. I would recommend showing how it can be inspected or visualized to derive meaningful insights to strengthen the contribution.
  - How does the approach scale to larger state spaces? I would like to request the authors to reflect on the memory consumption with respect to the graph size?
  - How much is the approach dependent on hand-crafted features? How does the model perform if noise is added to these features?
  - Could the authors report their results using standard benchmarking protocols for direct comparison with deep reinforcement learning methods?

  - there isn't enough context in the abstract when authors say "A new interpretable experimental learning model based on state history and global feedback is presented." in regards to what topic, what is being discussed here? Is this a RL model, is it a ML model, What is experimental learning?
  - overall the presentation is poor and I find quite enough details left out of the writing, which makes it harder for reader to understand the model being presented.
  - I also find it hard to parse what exactly is the goal here, as the experiments proceed in multiple stages and the presentation makes it hard to fully communicate the bigger picture and what this model is able to do. For e.g. in Sec 4.1 if the automated agent knows the rules of the game, what is it learning? is it learning the transition utility ? Is that nothing but the value function based on feedback i.e. reward?
  - How exactly is the state transformer doing the following "which was a multidimensional grayscale representation of a pixel map, and transform it into an interpretable representation of objects and events specific to the environment." ? What data does it need to transform the pixel spaces to such an interpretable representation? How is this data generated?
  - How is the historical memory of such state vectors as a sequence of T states maintained? Does this approach scale for large state spaces? If so, how? - - How is each node in the graph represented?
  - From all the figures and results, it is still not clear to me how this model adds interpretability?
  - I find this as a limitation to having to use "a custom simple computer vision algorithm that identified pixel regions on the game field that were distinct from the background, - which algorithm is being used? This process seems fairly manual.
  - The authors also acknowledge that their experimental methodology might be incomplete, as it was based on the number of games rather than steps. I find this a strong limiting point and needs to be addressed to support the validity of the method I believe, I am curious to hear if the authors believe otherwise.

  - There are many works for Interpretable RL that should be introduced in the related work.
  - Just using one environment cannot prove the advantages of the proposed method.
  - What does "playing 100 games" mean? How many environments are used in the experiment?
  - The colors of the different curves in Figure 3 seem to represent the same algorithm.
  - What is the definition of U? Why does the new method maximize U instead of Q function?


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

