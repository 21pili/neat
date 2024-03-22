import numpy as np
import pygame
from brain.neat import NeatAlgorithm
from game.game import Game, GameGraphics
from game.grid import Grid


def map_outputs(output, dt):
    """
    Map the output from the NEAT network to the acceleration and steering values (between -1 and 1)
    
    Args:
        output: output values from the NEAT network : size = 4 (accelerate, brake, left, right)
        dt: time step for the simulation
    """
    acc = dt * (np.exp(output[0]) - np.exp(output[1])) / (np.exp(output[0]) + np.exp(output[1]))
    steer = dt * (np.exp(output[2]) - np.exp(output[3])) / (np.exp(output[2]) + np.exp(output[3]))
    return acc, steer


def eval_genomes(genomes, current_config):
    """
    Simulate each genome, evaluate the fitness and update the population
    
    Args:
        genomes: list of tuples (genome_id, genome_instance) to evaluate
    """
    # Set the current neat configuration
    neat.config = current_config
    
    # Run each genome
    print("Evaluating new generation of genomes.")
    for i, (_, genome) in enumerate(genomes):
        # Log the current progress
        print(" -> Current progress : " + str(round(i/len(genomes) * 100)) + "%.", end="\r")
        
        # Create a new game instance for the current player
        game = Game(grid)
        
        # Run the game for each player
        elapsed_time = 0.0
        while not game.game_over and elapsed_time < PLAYER_MAX_TIME:
            # Get the inputs of the current game state
            inputs = game.get_inputs(PLAYER_RAY_COUNT)
            
            # Get the outputs from the neat network
            outputs = neat.predict(genome, inputs)
            
            # Execute the action on the game
            acc, steer = map_outputs(outputs, game.dt)
            game.update(acc, steer)
            
            # Update the elapsed time
            elapsed_time += game.dt
            
        # Compute fitness for the players
        fitness_params = game.get_fitness_parameters()
        genome.fitness = fitness_params[0]
        
    # Mutations, speciation, crossover, etc.
    # TODO


if __name__ == '__main__':
    # Run with graphics
    GAME_GRAPHICS = False
    
    # Simulation parameters
    GENERATIONS = 10        # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum lifetime of a player simulation
    PLAYER_RAY_COUNT = 5    # Number of rays to cast from the player
    
    # Load the circuit
    grid = Grid(250, 'circuit.png')

    if GAME_GRAPHICS:
        # Create the player
        player = GameGraphics(grid)
        
        # Run the game for the player
        while not player.game_over:
            player.update()
    
        # Quit the window
        pygame.quit()
    else:
        # Create and run the NEAT algorithm
        neat = NeatAlgorithm('brain/config.txt')
        neat.run(eval_genomes, GENERATIONS)
    
    