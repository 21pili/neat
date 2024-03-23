import numpy as np
from brain.neat import NeatAlgorithm
from game.game import Game
from game.grid import Grid
from multiprocessing import Pool


def map_outputs(output, dt):
    """
    Map the output from the NEAT network to the acceleration and steering values (between -1 and 1)
    
    Args:
        output: output values from the NEAT network : size = 4 (accelerate, brake, left, right)
        dt: time step for the simulation
    """
    acc = np.exp(output[0]) * 2 - 1
    steer = np.exp(output[1]) * 2 - 1
    if -0.5 < acc < 0.5:
        acc = 0
    
    return dt * acc, dt * steer

def run_simulation(args):
    """
    Run the simulation for a player and return the fitness
    
    Args:
        args: tuple (game, (network, (PLAYER_MAX_TIME, PLAYER_RAY_COUNT)))
    """
    # Get the arguments
    game, (network, (PLAYER_MAX_TIME, PLAYER_RAY_COUNT)) = args
    
    # Run the game for each player
    elapsed_time = 0.0
    while not game.game_over and elapsed_time < PLAYER_MAX_TIME:
        # Get the inputs of the current game state
        inputs = game.get_inputs(PLAYER_RAY_COUNT)
        
        # Get the outputs from the neat network
        outputs = network.activate(inputs)
        
        # Execute the action on the game
        acc, steer = map_outputs(outputs, game.dt)
        print(outputs, acc, steer)
        game.update(acc, steer)
        
        # Update the elapsed time
        elapsed_time += game.dt
        
    # Compute fitness for the players
    fitness_params = game.get_fitness_parameters()
    return fitness_params[0]

def eval_genomes(genomes, current_config):
    """
    Simulate each genome, evaluate the fitness and update the population
    
    Args:
        genomes: list of tuples (genome_id, genome_instance) to evaluate
    """
    # Set the current neat configuration
    neat.config = current_config
    
    # Run each genome
    with Pool() as pool:
        # Create the game instances and the networks
        games = [Game(grid) for _ in range(len(genomes))]
        neat_networks = [neat.create_network(genome) for _, genome in genomes]
        
        # Evaluate the fitness of each genome
        args = zip(games, zip(neat_networks, [(PLAYER_MAX_TIME, PLAYER_RAY_COUNT)] * len(genomes)))
        fitnesses = pool.map(run_simulation, args)
        
        # Set the fitness of each genome
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = fitnesses[i]


if __name__ == '__main__':
    # Run with graphics
    GAME_GRAPHICS = False
    
    # Simulation parameters
    GENERATIONS = 10        # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum lifetime of a player simulation
    PLAYER_RAY_COUNT = 10   # Number of rays to cast from the player
    
    # Load the circuit
    grid = Grid(250, 'circuit.png')

    if GAME_GRAPHICS:
        import pygame
        from game.game_graphics import GameGraphics
        
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
    
    