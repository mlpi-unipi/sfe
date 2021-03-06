# sfe
Stigmergy Flocking Evolution: adaptive exploration of robot swarms for distributed targets detection and tracking

![snapshot](https://github.com/mlpi-unipi/sfe/blob/master/snapshot.png)

Target search aims to discover elements of various complexity in a physical environment, with potential obstacles and no prior information on positions and layout. For example, illegal dumps in rural zones, mines in post-conflict lands, early wildfires in forests, humans trapped by earthquake in urban regions, early gas leaks in large industrial plants. The best coordination of the search process is achieved by minimizing the overall discovery time. In the literature, the capabilities of swarm robots for target search are largely acknowledged. For this purpose, different swarm intelligence algorithms have been proposed, inspired by biological species such as ants, fireflies, bees, birds, and so on. Despite the success of bio-inspired techniques (bio-heuristics), there are relevant algorithm selection and parametrization costs associated to every new type of mission or new instances of known missions. In order to automate the tuning of bio-inspired coordination for target search, in this paper a different approach, based on evolutionary optimization, is proposed. Experimental results on real-world scenarios clearly show a significant improvement of the mission performance after optimization. Although adaptive, the logic of bio-heuristics is nevertheless constrained by models of biological species, and then it is somehow holistic. As a such, it cannot be modularized nor aggregated. To overcome this limits, a new design approach based on hyper-heuristics is proposed. It is a search methodology that automates the combination of simple heuristics to generate more adaptable logics. The approach consists in an optimization method of modular heuristics for target search, whose aggregation and tuning are represented on a unique and continuous search space. As a result, an efficient heuristics hybridization is generated for a given application domain, in which an evolutionary optimization minimizes the discovery time. A modeling and optimization testbed has been developed and publicly released. Experimental results on real-world scenarios show that a hyper-heuristic based on stigmergy and flocking significantly outperforms the adaptive bio-heuristics.

## References

<a id="1">[1]</a>
Jaxa-Rozen & Kwakkel (2018).
PyNetLogo: Linking NetLogo with Python.
Journal of Artificial Societies and Social Simulation, 21 (2) 4.
http://jasss.soc.surrey.ac.uk/21/2/4.html.
DOI: 10.18564/jasss.3668.

<a id="2">[2]</a>
Chathika Gunaratne and Ivan Garibay (2018).
NL4Py: Agent-Based Modeling in Python with Parallelizable NetLogo Workspaces.
arXiv preprint arXiv:1808.03292.

<a id="3">[3]</a>
Gunaratne, C. (2018).
NL4Py.
https://github.com/chathika/NL4Py.
Complex Adaptive Systems Lab, University of Central Florida, Orlando, FL.

<a id="4">[4]</a>
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html.
