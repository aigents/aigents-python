# State-based History-aware Artificial Reinforcement Intelligence Kernel (Sharik) 1.2 - TODO

## TODO NEXT
- REVIEW
  https://paperreview.ai/review?token=wrkiuc22TFMIZkGEwX6u7r62vF1xq85Wl_0AKDwfzVM
  https://arxiv.org/pdf/2211.15183 (Continuous Episodic Control - TODO)
- IMPROVE
  - dimensionatilty reduction
    - pixel map C(X,Y,RGB) => grayscale c or JPEG-style!?
    - grayscale G(X,Y) => spots "spot transformation" ... !!!???
      - smooth (optional, for cross-frame similarity)?
      - cutoff triples by theshshold to binary map BW(X,Y)
      - convert binary pixel map BW(X,Y) to graph of triples (G,X,Y)
      - cluster (G,X,Y) till they can be clustered
    - as coefficient codes ... !!!???
    - to object map (Object,Property)
  - learning stability
    - ! make sure that dU applies to transtions in a sequential epizode, not a states in it! 
    - c - chance to perform random action ("curiosity"), make dependant on "surpriziness" z_t = (s_t - s't)/max(s_t,s't), as c_t = Z * z_t
    - g - extra positive feedback for predictiviness (1 - z_t), as d_t = R * (1-z_t), where R can be considered as an element of x
      OR
      - the same for "discoverinness"?
    - e - extra negative feedback for energy consumption

- ICDM sumbission
  - http://icdm2026.neu.edu.cn/ 10 pages
    - Paper    June 6, 2026
      - https://anonymous.4open.science/dashboard
    - Accept   August 16, 2026
    - Camera   September 9, 2026
    - Conf.    November 12-15, 2026

  - Topics
    - Machine learning, deep learning, and statistical methods for big data.
    - Mining heterogeneous data sources, including text, semi-structured, spatio-temporal, streaming, graph, web, and multimedia data
  - Checklist
    Yes 1.1 A clear description of the mathematical setting, algorithm, and/or model.
    Yes No NA	1.2 A clear explanation of any assumptions.
    Yes No NA	1.3 An analysis of the complexity (time, space, sample size) of any algorithm.
    Q2. For any theoretical claim, check if you include:
    NA	2.1 A clear statement of the claim.
    NA	2.2 A complete proof of the claim.
    Q3. For all datasets used, check if you include:
    Yes 3.1 The relevant statistics, such as number of examples.
    Yes 3.2 The details of train/validation/test splits.
    NA	3.3 An explanation of any data that were excluded, and all pre-processing step.
    Yes 3.4 A link to a downloadable version of the dataset or simulation environment.
    NA	3.5 For new data collected, a complete description of the data collection process, such as instructions to annotators and methods for quality control.
    Q4. For all shared code related to this work, check if you include:
    Yes	4.1 Specification of dependencies.
    Yes 4.2 Training code.
    Yes 4.3 Evaluation code.
    NA	4.4 (Pre-)trained model(s).
    Yes 4.5 README file includes table of results accompanied by precise command to run to produce those results.
    Q5. For all reported experimental results, check if you include:
    Yes 5.1 The range of hyper-parameters considered, method to select the best hyper-parameter configuration, and specification of all hyper-parameters used to generate results.
    Yes 5.2 The exact number of training and evaluation runs.
    Yes 5.3 A clear definition of the specific measure or statistics used to report results.
    Yes 5.4 A description of results with central tendency (e.g.mean) & variation (e.g. error bars).
    Yes 5.5 The average runtime for each result, or estimated energy cost.
    Yes	5.6 A description of the computing infrastructure used.


- Summary on review-based improvements
  - improve paper
    + mention process mining
    + cleaner abstract and body - what is experiential learning
    + cite more interpretable RL papers
    - ! checklist
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

