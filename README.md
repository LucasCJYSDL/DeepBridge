# DeepBridge
Codebase for an unfinished projects due to the restriction of computation resources.

Language: Python

The following components are included:
- A Bridge simulator written with Python, which is connected to a double-dummy solver to calculate playing scores and a Bridge program for visulization.
- Implementations of SOTA MARL algorithms: MAVEN, QMIX, Weighted QMIX, MSAC, COMA, which are used as the main training algorithms and baselines.
- Learning by self-playing: two agents are trained as a team by competing with another team.
- Heuristic search to aid the exploration and learning efficiency of the agents.
- Policy initialization through imitation learning based on human expert boarding data.
