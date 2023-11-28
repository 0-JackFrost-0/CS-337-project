# CS-337-project

This repository contains the code for our CS 337 project - **Deep Ensemble Reinforcement Learning for Adaptive
Trading**. The team members are: Shubham Hazra (210100143), Om Godage (21d100006) and 
Vijay Balsubramaniam (21d180043). 
The project focuses on the application of Deep Reinforcement Learning for trading and 
explores the use of ensemble methods to improve the performance of the agent. We also 
compare the performance of the agent with the baseline index fund and statistical algorithms
like HRP and CVaR under Modern Portfolio Theory.

## Installation

The code is written in Python 3.10. The required packages are listed in `requirements.txt`.
To install the packages, run the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure

There are 3 main directories in the repository:

1. `./`: Contains the report (report.pdf) and the requirements file.
2. `src`: Contains the source code for the project.
3. `results`: Contains the results of the experiments.
4. `plots`: Contains the plots of the results.

## Running the code

The code files are present in the `src` directory. 
The three main files are:
1. `ensemble.ipynb`: Contains the code for the ensemble methods.
2. `MPT.py`: Contains the code for the statistical algorithms. 
3. `plot.ipynb`: Contains the code to plot the results.

You can just run the cells in the notebook to reproduce the results. The results are stored in the `results` directory and the plots are stored in the `plots` directory.

## References
[FinRL](https://github.com/AI4Finance-Foundation/FinRL)
