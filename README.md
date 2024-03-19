# A Comparative Study of Simulated and Quantum Annealing for the Travelling Salesman Problem

Read the full report [here](report.pdf)!

## Abstract

Solving combinatorial optimisation (CO) problems efficiently is an important challenge that spans various
domains. In this report, we investigate and compare two prominent meta-heuristic approaches to tackle CO
problems: simulated annealing (SA) and quantum annealing (QA). We use a 16-city instance of the Travelling
Salesman Problem (TSP) as a test example and solve it via standard Monte Carlo SA and path-integral Monte
Carlo (PIMC) QA. We determine the range of initial temperature $T_0$ for SA, as well as the ranges of Trotter
number $P$ and ambient temperature $T$ for QA, that yield the most efficient and accurate results through studies of the dependence of the final residual length on each method’s parameters. The superiority of PIMC QA relative to SA is demonstrated, for this TSP, with a study of the dependence of the final residual length on the total number of Monte Carlo steps. These results suggest QA could be an even better general-purpose CO meta-heuristic than SA. Future areas of investigation for QA and quantum optimisation as a whole are suggested.
