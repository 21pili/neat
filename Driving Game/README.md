## **Introduction**
This project aims to study the NEAT (NeuroEvolution of Augmenting Topologies) algorithm on the problem of a racing game.  The racing game is a simple 2D game where the player controls a car on a looping track, while seeing with raycasts the distance to the walls. The goal is to complete the track as fast as possible. The networks are stored in pickle files, and the fitness data in csv files. There is also a jupityer notebook that can be used to visualize the data.

## **How to run this project**
### Prerequisites
- Clone this repository and enter the python environment.
- Install the required packages by running the following command:
```bash
pip install -r requirements.txt
```
- To change the parameters used by the NEAT algorithm, edit the `config.txt` file in the `brain/` folder.

### Modify the simulation mode
- Open `main.py` file and change the value of the variables to one of the following values:
  - GAME_GRAPHICS (Bool) : run the simulation with graphics
  - LOAD_CHECKPOINT (Bool) : load an existing neural network instead of training a new one

### Run the simulation
- Finally, run the `main.py` file. The simulation will start and the results will be saved in the `checkpoints/` folder. This folder will contain the neural networks and the fitness data in pickle and csv files, respectively.
- If the simulation is run with graphics, the player can control the car with the arrow keys. The AI will play the game if the simulation is run without graphics, without saving the results.
- To visualize the results of a training session, run the `plots.ipynb` jupyter notebook.

## **References**
Main sources used for this project:
- Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.
- Stanley, K. O., & Bryant, B. D. (2005). Real-time neuroevolution in the NERO video game. Evolutionary Computation, 2005. The 2005 IEEE Congress on, 1879-1886.