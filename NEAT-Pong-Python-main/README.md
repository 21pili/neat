# An implementation of rt-neat


This project is an attempt to implement rt-neat described in the references below on the Pong game using the neat package. Real-Time Neat is a version of neat where agents in a game are evoluting while the game is being played. Here, we train a population of neural networks and replace the worst performing agent by a crossover of the best fit agents every $m$ seconds.

## How to run this project

### Prerequisites

Clone this repository and enter the python environment.
Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
To change the parameters used by the NEAT algorithm, edit the config.txt file in the brain/ folder.

The code is in the file test_rt_neat.py. Run this python file to run the evolution algorithm and then play against the winner. Good luck !

The other files are saving files used by the algorithm.

### Main sources used for this project:

Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.
Stanley, K. O., & Bryant, B. D. (2005). Real-time neuroevolution in the NERO video game. Evolutionary Computation, 2005. The 2005 IEEE Congress on, 1879-1886.

