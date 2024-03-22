import pygame
import numpy as np
import os
import neat
from game.game import Game, GameGraphics
from game.grid import Grid
import tools.population as pop

# # Run the game with graphics
# if __name__ == "__main__":
#     # Load the circuit
#     grid = Grid(250, 'circuit.png')
    
#     # Create the player
#     player = GameGraphics(grid)
        
#     # Run the game for each player
#     while not player.game_over:
#         # Execute the action
#         player.update()
    
#     # Wait for 3 seconds before quitting
#     pygame.time.wait(3000)
    
#     # Quit the window
#     pygame.quit()


def convert_output(output,dt):
    """
    output size : 4 (accélerer, freiner, gauche, droite) <-> (Z, S, Q, D)
    Convert the output from the NEAT network to the acceleration and 
    steering values (between -1 and 1)
    """
    acc = dt * (np.exp(output[0]) - np.exp(output[1])) / (np.exp(output[0]) + np.exp(output[1]))
    steer = dt * (np.exp(output[2]) - np.exp(output[3])) / (np.exp(output[2]) + np.exp(output[3]))
    
    # print('acc :', acc, '     steer :', steer)
    return acc, steer

# Run the game with the NEAT algorithm

def eval_genomes(genomes, config):
    """
    Run each genome against eachother one time to determine the fitness.
    """
    grid = Grid(250, 'circuit.png')
    
    for i, (genome_id, genome) in enumerate(genomes):
        print(round(i/len(genomes) * 100), end=" ")
        game = Game(grid)
        
        # Run the game for each player
        elapsed_time = 0.0
        while not game.game_over and elapsed_time < PLAYER_MAX_TIME:
            
            # Get the inputs to the neat network
            # (vel_x, vel_y, acc, steer, rot, ray_distance_1, ..., ray_distance_n)
            inputs = game.get_inputs(PLAYER_RAY_COUNT) 
            print('inputs :', inputs)
            
            # Get the output from the neat network
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            output = net.activate(inputs)
            
            # Execute the action 
            # (forward, backward, left, right) -> (acc, steer)
            acc, steer = convert_output(output, game.dt)
            game.update(acc, steer)
            
            # Update the elapsed time
            elapsed_time += game.dt
            
        # Compute fitness for the players
        fitness_params = game.get_fitness_parameters()
        genome.fitness = fitness_params[0]
        
    # Mutations, speciation, crossover, etc.


if __name__ == '__main__':
    
    # Algorithm parameters
    GENERATIONS = 10 # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum time for a player to run
    PLAYER_RAY_COUNT = 5 # Number of rays to cast
    
    local_dir = os.path.dirname(__file__)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         os.path.join(local_dir, 'tools/config.txt'))
    
    p = pop.create_population(config)
    p.run(eval_genomes, GENERATIONS)
    
    