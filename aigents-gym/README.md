# State-based History-aware Artificiall Reinforcement Intelligence Kernel (Sharik) - Files, Classes, TODO and Diary 

## Files

- [breakout_hack_simple.py](./breakout_hack_simple.py) - "cheating" Breakout player, knows rules of the game and properties of the game field, wins 860 (out of top 864 points) always if not limited by number of steps per game
- [breakout_hack_simple1.py](./breakout_hack_simple1.py) - same as above logic wrapped in BreakoutHacky

## Classes

- player.BreakoutHacky - "cheating" Breakout player, knows rules of the game and properties of the game field
  - if action = env.action_space.sample() at game (re)start: wins 831/860 (out of top 864 points, varies even with fixed random seed) if not limited by number of steps per game (108000 steps max) or 616/725/732/856 (varies even with fixed random seed) if limited by 18000 steps
  - if action = 1 at game (re)start: wins 860 (out of top 864 points) always (regardless of random seed) if not limited by number of steps per game (108000 steps max) or 732 (regardless of random seed) if limited by 18000 steps
- ...

## DONE
- 2026-01-09/.../16 Experiments with different random seeds (S) and state similarities (SS), selected: 41 (best), 3 (average), 2 (worst)
- 2026-01-09/.../16 Experiment with hyper-parameters for the best seed (41), find the best: LM=2, TU=0, CS=2, TC=1, SC=2, SR=True, SS=0.9/0.95, CU=False, EA=False, 

## TODO
- 2026-01-13/... run up to 5K games with selected seeds for SS=0.9 and SS=0.95, to collect the difference (make sure the performance is seed-agnostic)
  - 41 (DONE)
  - 3 (PROGRESS)
  - 2 
- draft paper of ICML https://icml.cc/ (PROGRESS)
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

- https://link.springer.com/chapter/10.1007/978-3-030-93758-4_12 Experiential Kolonin
- https://arxiv.org/abs/2509.07009 Psyche Kolonin
- https://arxiv.org/abs/2512.10985 Marti
- https://arxiv.org/abs/2511.22130 Experiential learning
- https://dl.acm.org/doi/abs/10.5555/2566972.2566979 Marc G. Bellemare, The arcade learning environment: an evaluation platform for general agents

- https://dl.acm.org/doi/10.1109/TPAMI.2013.50 Yoshua Bengio, Representation Learning: A Review and New Perspectives
@article{10.1109/TPAMI.2013.50,
author = {Bengio, Yoshua and Courville, Aaron and Vincent, Pascal},
title = {Representation Learning: A Review and New Perspectives},
year = {2013},
issue_date = {August 2013},
publisher = {IEEE Computer Society},
address = {USA},
volume = {35},
number = {8},
issn = {0162-8828},
url = {https://doi.org/10.1109/TPAMI.2013.50},
doi = {10.1109/TPAMI.2013.50},
abstract = {The success of machine learning algorithms generally depends on data representation, and we hypothesize that this is because different representations can entangle and hide more or less the different explanatory factors of variation behind the data. Although specific domain knowledge can be used to help design representations, learning with generic priors can also be used, and the quest for AI is motivating the design of more powerful representation-learning algorithms implementing such priors. This paper reviews recent work in the area of unsupervised feature learning and deep learning, covering advances in probabilistic models, autoencoders, manifold learning, and deep networks. This motivates longer term unanswered questions about the appropriate objectives for learning good representations, for computing representations (i.e., inference), and the geometrical connections between representation learning, density estimation, and manifold learning.},
journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
month = aug,
pages = {1798–1828},
numpages = {31},
keywords = {unsupervised learning, representation learning, neural nets, feature learning, autoencoder, Speech recognition, Neural networks, Manifolds, Machine learning, Learning systems, Feature extraction, Deep learning, Boltzmann machine, Abstracts}
}

- https://dl.acm.org/doi/10.5555/3454287.3455074 Yoshua Bengio, Unsupervised state representation learning in atari
@inbook{10.5555/3454287.3455074,
author = {Anand, Ankesh and Racah, Evan and Ozair, Sherjil and Bengio, Yoshua and C\^{o}t\'{e}, Marc-Alexandre and Hjelm, R. Devon},
title = {Unsupervised state representation learning in atari},
year = {2019},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
abstract = {State representation learning, or the ability to capture latent generative factors of an environment, is crucial for building intelligent agents that can perform a wide variety of tasks. Learning such representations without supervision from rewards is a challenging open problem. We introduce a method that learns state representations by maximizing mutual information across spatially and temporally distinct features of a neural encoder of the observations. We also introduce a new benchmark based on Atari 2600 games where we evaluate representations based on how well they capture the ground truth state variables. We believe this new framework for evaluating representation learning models will be crucial for future representation learning research. Finally, we compare our technique with other state-of-the-art generative and contrastive representation learning methods. The code associated with this work is available at https://github.com/mila-iqia/atari-representation-learning},
booktitle = {Proceedings of the 33rd International Conference on Neural Information Processing Systems},
articleno = {787},
numpages = {14}
}

Industrial Automation

https://www.mdpi.com/2075-1702/13/12/1140
@Article{machines13121140,
AUTHOR = {Alginahi, Yasser M. and Sabri, Omar and Said, Wael},
TITLE = {Reinforcement Learning for Industrial Automation: A Comprehensive Review of Adaptive Control and Decision-Making in Smart Factories},
JOURNAL = {Machines},
VOLUME = {13},
YEAR = {2025},
NUMBER = {12},
ARTICLE-NUMBER = {1140},
URL = {https://www.mdpi.com/2075-1702/13/12/1140},
ISSN = {2075-1702},
ABSTRACT = {The accelerating integration of Artificial Intelligence (AI) in Industrial Automation has established Reinforcement Learning (RL) as a transformative paradigm for adaptive control, intelligent optimization, and autonomous decision-making in smart factories. Despite the growing literature, existing reviews often emphasize algorithmic performance or domain-specific applications, neglecting broader links between methodological evolution, technological maturity, and industrial readiness. To address this gap, this study presents a bibliometric review mapping the development of RL and Deep Reinforcement Learning (DRL) research in Industrial Automation and robotics. Following the PRISMA 2020 protocol to guide the data collection procedures and inclusion criteria, 672 peer-reviewed journal articles published between 2017 and 2026 were retrieved from Scopus, ensuring high-quality, interdisciplinary coverage. Quantitative bibliometric analyses were conducted in R using Bibliometrix and Biblioshiny, including co-authorship, co-citation, keyword co-occurrence, and thematic network analyses, to reveal collaboration patterns, influential works, and emerging research trends. Results indicate that 42% of studies employed DRL, 27% focused on Multi-Agent RL (MARL), and 31% relied on classical RL, with applications concentrated in robotic control (33%), process optimization (28%), and predictive maintenance (19%). However, only 22% of the studies reported real-world or pilot implementations, highlighting persistent challenges in scalability, safety validation, interpretability, and deployment readiness. By integrating a review with bibliometric mapping, this study provides a comprehensive taxonomy and a strategic roadmap linking theoretical RL research with practical industrial applications. This roadmap is structured across four critical dimensions: (1) Algorithmic Development (e.g., safe, explainable, and data-efficient RL), (2) Integration Technologies (e.g., digital twins and IoT), (3) Validation Maturity (from simulation to real-world pilots), and (4) Human-Centricity (addressing trust, collaboration, and workforce transition). These insights can guide researchers, engineers, and policymakers in developing scalable, safe, and human-centric RL solutions, prioritizing research directions, and informing the implementation of Industry 5.0–aligned intelligent automation systems emphasizing transparency, sustainability, and operational resilience.},
DOI = {10.3390/machines13121140}
}

https://ieeexplore.ieee.org/document/9619637
@INPROCEEDINGS{9619637,
  author={Xin, Quan and Wu, Guanlin and Fang, Wenqi and Cao, Jiang and Ping, Yang},
  booktitle={2021 7th International Conference on Big Data and Information Analytics (BigDIA)}, 
  title={Opportunities for Reinforcement Learning in Industrial Automation}, 
  year={2021},
  volume={},
  number={},
  pages={496-504},
  keywords={Manufacturing industries;Technological innovation;Automation;Costs;Process control;Reinforcement learning;Companies;Reinforcement learning;artificial intelligence;industrial automation},
  doi={10.1109/BigDIA53151.2021.9619637}}

Home ("Smart Home") Automation

https://ieeexplore.ieee.org/document/10724603
@INPROCEEDINGS{10724603,
  author={Sen, Amit Prakash and Goyal, Manish Kumar and Shalini},
  booktitle={2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT)}, 
  title={Deploying Reinforcement Learning Approaches for Smart Home Automation}, 
  year={2024},
  volume={},
  number={},
  pages={1-5},
  keywords={Electric potential;Automation;Green products;Smart homes;Reinforcement learning;Turning;User experience;Security;Reliability;Protection;Distinctive;Methodological;Protection;Domestic;Deployment;Potential;Reinforcement},
  doi={10.1109/ICCCNT61001.2024.10724603}}

https://www.mdpi.com/1996-1073/17/24/6420
@Article{en17246420,
AUTHOR = {Latoń, Dominik and Grela, Jakub and Ożadowicz, Andrzej},
TITLE = {Applications of Deep Reinforcement Learning for Home Energy Management Systems: A Review},
JOURNAL = {Energies},
VOLUME = {17},
YEAR = {2024},
NUMBER = {24},
ARTICLE-NUMBER = {6420},
URL = {https://www.mdpi.com/1996-1073/17/24/6420},
ISSN = {1996-1073},
ABSTRACT = {In the context of the increasing integration of renewable energy sources (RES) and smart devices in domestic applications, the implementation of Home Energy Management Systems (HEMS) is becoming a pivotal factor in optimizing energy usage and reducing costs. This review examines the role of reinforcement learning (RL) in the advancement of HEMS, presenting it as a powerful tool for the adaptive management of complex, real-time energy demands. This review is notable for its comprehensive examination of the applications of RL-based methods and tools in HEMS, which encompasses demand response, load scheduling, and renewable energy integration. Furthermore, the integration of RL within distributed automation and Internet of Things (IoT) frameworks is emphasized in the review as a means of facilitating autonomous, data-driven control. Despite the considerable potential of this approach, the authors identify a number of challenges that require further investigation, including the need for robust data security and scalable solutions. It is recommended that future research place greater emphasis on real applications and case studies, with the objective of bridging the gap between theoretical models and practical implementations. The objective is to achieve resilient and secure energy management in residential and prosumer buildings, particularly within local microgrids.},
DOI = {10.3390/en17246420}
}

https://dl.acm.org/doi/10.1145/3642975.3678961
@inproceedings{10.1145/3642975.3678961,
author = {Christopoulos, Marios and Spantideas, Sotirios and Giannopoulos, Anastasios and Trakadas, Panagiotis},
title = {Deep Reinforcement Learning for Smart Home Temperature Comfort in IoT-Edge Computing Systems},
year = {2024},
isbn = {9798400705434},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3642975.3678961},
doi = {10.1145/3642975.3678961},
abstract = {In this paper, a novel IoT-Edge-Cloud (IEC) computing system designed for multiple Smart Homes is introduced, with a focus on supporting Home Energy Management Systems (HEMS) for temperature control within a defined comfort range. Leveraging model-free deep reinforcement learning, the proposed method, Smart Home Energy and Temperature Control (SHETEC), employs autonomous agents which are trained to manipulate the input power of Heating, Ventilation, and Air Conditioning (HVAC) systems and charging/discharging power of Energy Storage Systems (ESS) using Deep Deterministic Policy Gradients (DDPG). In addition, we present the Average Opinion (AO) method, a collaborative decision-making approach that combines the models of all Smart Homes in a distributed approach. Experimental results, conducted through simulation on three Smart Homes using real-world heterogeneous data, demonstrate the effectiveness of both SHETEC and Average Opinion in maintaining temperatures within the desired comfort bounds.},
booktitle = {Proceedings of the 1st International Workshop on MetaOS for the Cloud-Edge-IoT Continuum},
pages = {1–7},
numpages = {7},
keywords = {Deep Reinforcement Learning (DRL), Energy Management, Heating Ventilation and Air Conditioning (HVAC) systems, IoT-Edge-Cloud Continuum (IEC), Smart Home, Temperature Comfort},
location = {Athens, Greece},
series = {MECC '24}
}

### Mission

https://arxiv.org/abs/2502.09417v1
"prevalent challenges encountered in RL optimization, including issues related to sample efficiency and scalability; safety and robustness; interpretability and trustworthiness"
@inproceedings{Farooq_2024,
   title={A Survey of Reinforcement Learning for Optimization in Automation},
   url={http://dx.doi.org/10.1109/CASE59546.2024.10711718},
   DOI={10.1109/case59546.2024.10711718},
   booktitle={2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)},
   publisher={IEEE},
   author={Farooq, Ahmad and Iqbal, Kamran},
   year={2024},
   month=aug, pages={2487–2494} }


### Prior Art

https://dl.acm.org/doi/abs/10.5555/2566972.2566979
Marc G. Bellemare, The arcade learning environment: an evaluation platform for general agents
"18000"!
@article{10.5555/2566972.2566979,
author = {Bellemare, Marc G. and Naddaf, Yavar and Veness, Joel and Bowling, Michael},
title = {The arcade learning environment: an evaluation platform for general agents},
year = {2013},
issue_date = {May 2013},
publisher = {AI Access Foundation},
address = {El Segundo, CA, USA},
volume = {47},
number = {1},
issn = {1076-9757},
abstract = {In this article we introduce the Arcade Learning Environment (ALE): both a challenge problem and a platform and methodology for evaluating the development of general, domain-independent AI technology. ALE provides an interface to hundreds of Atari 2600 game environments, each one different, interesting, and designed to be a challenge for human players. ALE presents significant research challenges for reinforcement learning, model learning, model-based planning, imitation learning, transfer learning, and intrinsic motivation. Most importantly, it provides a rigorous testbed for evaluating and comparing approaches to these problems. We illustrate the promise of ALE by developing and benchmarking domain-independent agents designed using well-established AI techniques for both reinforcement learning and planning. In doing so, we also propose an evaluation methodology made possible by ALE, reporting empirical results on over 55 different games. All of the software, including the benchmark agents, is publicly available.},
journal = {J. Artif. Int. Res.},
month = may,
pages = {253–279},
numpages = {27}
}

https://arxiv.org/abs/1908.04683
@misc{toromanoff2019deepreinforcementlearningreally,
      title={Is Deep Reinforcement Learning Really Superhuman on Atari? Leveling the playing field}, 
      author={Marin Toromanoff and Emilie Wirbel and Fabien Moutarde},
      year={2019},
      eprint={1908.04683},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/1908.04683}, 
}

https://arxiv.org/pdf/1312.5602
"The statistics were computed by running an -greedy policy with  =
0.05 for 10000 steps. The two plots on the right show the average maximum predicted action-value
of a held out set of states on Breakout and Seaquest respectively. One epoch corresponds to 50000
minibatch weight updates or roughly 30 minutes of training time."
@misc{mnih2013playingatarideepreinforcement,
      title={Playing Atari with Deep Reinforcement Learning}, 
      author={Volodymyr Mnih and Koray Kavukcuoglu and David Silver and Alex Graves and Ioannis Antonoglou and Daan Wierstra and Martin Riedmiller},
      year={2013},
      eprint={1312.5602},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1312.5602}, 
}


https://arxiv.org/pdf/1908.04683 => 5 min (18000 frames)
https://arxiv.org/pdf/1312.5602 => One epoch corresponds to 50000 minibatch weight updates or roughly 30 minutes of training time.
=> 1 epoch in Mnih, 2013 = 30 minutes = 6 X 5 minutes = 108000 frames == capped max frames per game!
=> 100 epochs in Mnih = 100 * 108000 = 10,800,000


https://arxiv.org/pdf/2003.13350
Agent57 "864!"
@misc{badia2020agent57outperformingatarihuman,
      title={Agent57: Outperforming the Atari Human Benchmark}, 
      author={Adrià Puigdomènech Badia and Bilal Piot and Steven Kapturowski and Pablo Sprechmann and Alex Vitvitskyi and Daniel Guo and Charles Blundell},
      year={2020},
      eprint={2003.13350},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2003.13350}, 
}

https://arxiv.org/abs/1911.08265
MuZero "864!" "pllanning, hidden state"
@article{Schrittwieser_2020,
   title={Mastering Atari, Go, chess and shogi by planning with a learned model},
   volume={588},
   ISSN={1476-4687},
   url={http://dx.doi.org/10.1038/s41586-020-03051-4},
   DOI={10.1038/s41586-020-03051-4},
   number={7839},
   journal={Nature},
   publisher={Springer Science and Business Media LLC},
   author={Schrittwieser, Julian and Antonoglou, Ioannis and Hubert, Thomas and Simonyan, Karen and Sifre, Laurent and Schmitt, Simon and Guez, Arthur and Lockhart, Edward and Hassabis, Demis and Graepel, Thore and Lillicrap, Timothy and Silver, David},
   year={2020},
   month=dec, pages={604–609} 
}




## Paper

We propse

Conclusion
Основные проблемы для решения:
1. Пространство состояний, включая сенсорные входы, выходы и потребности, включая избегание наказания, получение подкрепления, совпадение ожиданий и сбережение ресурсов
2. Понижение размерности состояний (включая сенсорные входы) он полного растра до 1-хот векторов выявленных объектов или их координат
3. Обучение с отложенным подкреплением, вплоть до за выигранный раунд или партию

### Keywords

### Impact Statement

Interpretable 

Key Terms & Concepts:
Edge Computing: Performing computations closer to the data source (the "edge" of the network) on small, often embedded devices, rather than sending everything to a central cloud.
Volunteer Computing: Using the idle processing power of personal computers (like through projects using BOINC) for large distributed tasks, effectively pooling weak resources.
Ubiquitous Computing/Pervasive Computing: The idea of tiny, cheap computers embedded everywhere in the environment, working together, which fits the description.
"Resource-Constrained Computing": A more general, descriptive term for systems with limited CPU, memory, or power.
"Low-End Computing": A simple, non-technical way to describe using less powerful hardware. 
