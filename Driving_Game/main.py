import numpy as np
from brain.neat import NeatAlgorithm
from game.game import Game
from game.grid import Grid
from multiprocessing import Pool


def map_outputs(output, dt, player):
    """
    Map the output from the NEAT network to the acceleration and steering values
    
    Args:
        output: output values from the NEAT network : size = 4 (accelerate, acceleration value, brake, steer)
        dt: time step for the simulation
    """
    # Neuron 0: Accelerate or not
    # Neuron 1: Value of acceleration between 0 and 1
    # Neuron 2: Brake or not
    acc = 0
    if output[0] >= 0.5:
        acc += output[1]
    if output[2] >= 0.5:
        acc -= player.brake_acc
    
    # Neuron 3: Steer value between -1 and 1 (from left to right)
    steer = output[3] * 2 - 1
    
    # Map the values to the player limits
    acc = dt * (acc * 2.0 * player.max_acc - player.max_acc) / player.acc_mult
    steer = dt * (steer * 2.0 * player.max_steer - player.max_steer) / player.steer_mult
    
    return acc, steer

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
        acc, steer = map_outputs(outputs, game.dt, game.player)
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
    # Configuration
    GAME_GRAPHICS = True
    LOAD_CHECKPOINT = True
    
    # File paths
    CONFIG_FILE = 'brain/config.txt'
    CHECKPOINT_FILE = 'checkpoints/best'
    
    # Simulation parameters
    GENERATIONS = 10        # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum lifetime of a player simulation
    PLAYER_RAY_COUNT = 10   # Number of rays to cast from the player
    
    # Load the circuit
    grid = Grid(250, 'circuit.png')

    if GAME_GRAPHICS:
        import pygame
        from game.game_graphics import GameGraphics
        
        if LOAD_CHECKPOINT:
            # Create and run the NEAT algorithm loading from a checkpoint
            neat = NeatAlgorithm(CONFIG_FILE, CHECKPOINT_FILE)
        
        # Create the game with graphics
        game = GameGraphics(grid)
        
        # Run the game for the player
        while not game.game_over:
            # Update the game state
            game.tick()
            
            # Handle window events
            game.events()
            
            if not LOAD_CHECKPOINT:
                # Get the inputs of the current game state
                acc, steer = game.key_inputs()
            else:
                # Get the inputs of the current game state
                inputs = game.get_inputs(PLAYER_RAY_COUNT)
                
                # Create the network from the genome
                net = neat.create_network(neat.best)
                
                # Get the outputs from the neat network
                outputs = net.activate(inputs)
                
                # Execute the action on the game
                acc, steer = map_outputs(outputs, game.dt, game.player)
            
            # Update the game state
            game.update(acc, steer)
            
            # Display the game
            game.draw()
    
        # Quit the window
        pygame.quit()
    else:
        # Create and run the NEAT algorithm
        neat = NeatAlgorithm(CONFIG_FILE)
        winner = neat.run(eval_genomes, GENERATIONS)
        
        # Save the best genome
        neat.save_genome(CHECKPOINT_FILE, winner)
    
    