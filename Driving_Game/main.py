import os
import time
import numpy as np
from brain.neat import NeatAlgorithm
from game.game import Game
from game.grid import Grid
from multiprocessing import Pool
from neat.six_util import iteritems
import warnings
import csv


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
    acc = (acc * 2.0 * player.max_acc - player.max_acc) / player.acc_mult
    if output[2] >= 0.5:
        acc -= player.brake_acc
    
    # Neuron 3: Steer value
    steer = (output[3] * 2.0 * player.max_steer - player.max_steer) / player.steer_mult
    
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
    
     # Check if out of time
    if elapsed_time >= PLAYER_MAX_TIME and fitness_params[0] > 0.1:
        warnings.warn('Agent out of time')
    return fitness_params[0]

def eval_genomes(genomes, current_config):
    """
    Simulate each genome, evaluate the fitness and update the population
    
    Args:
        genomes: list of tuples (genome_id, genome_instance) to evaluate
    """
    global grid, PLAYER_POS, map_name
    
    # Set the current neat configuration
    neat.config = current_config
    generation = neat.population.generation
    
    if np.random.rand() < PROB_CHANGE_MAP:
        grid, PLAYER_POS, map_name = load_map(MAP_FOLDER)

    # Run each genome
    with Pool() as pool:
        # Create the game instances and the networks
        games = [Game(grid, PLAYER_POS, DT) for _ in range(len(genomes))]
        neat_networks = [neat.create_network(genome) for _, genome in genomes]
        # Evaluate the fitness of each genome
        args = zip(games, zip(neat_networks, [(PLAYER_MAX_TIME, PLAYER_RAY_COUNT)] * len(genomes)))
        fitnesses = pool.map(run_simulation, args)
        
        # Evaluate the fitness on a single map for benchmarking
        grid_bench, PLAYER_POS_bench, _ = load_map('maps/circuit_paillon/')
        games_bench = [Game(grid_bench, PLAYER_POS_bench, DT) for _ in range(len(genomes))]
        args_bench = zip(games_bench, zip(neat_networks, [(PLAYER_MAX_TIME, PLAYER_RAY_COUNT)] * len(genomes)))
        fitnesses_bench = pool.map(run_simulation, args_bench)
        
        # Set the fitness of each genome as benchmark value to save
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = fitnesses_bench[i] * 10.0 if BENCHMARK_PAILLON else fitnesses[i] * 10.0
        
        # Save the species statistics as (generation, specie_id, agent_id, fitness) lines in a csv file
        save_species_statistics(neat.population.species.species, generation)
        
        # save all genomes
        if SAVE_BEST_GENOME_ONLY:
            os.makedirs(SAVING_FOLDER, exist_ok=True)
            best = np.argmax(fitnesses)
            name_save = SAVING_FOLDER + 'gen{}-fit{:.3f}-'.format(generation, genome.fitness) + map_name[8:]
            neat.save_genome(name_save, genomes[best][1])
        else:
            gen_folder = SAVING_FOLDER + 'gen{}-'.format(generation)  + map_name[8:] + '/'
            os.makedirs(gen_folder, exist_ok=True)
            for i, (genome_id, genome) in enumerate(genomes):
                neat.save_genome(gen_folder + 'id{}-fit{:.3f}'.format(genome_id, genome.fitness), genome)
        
        # Correct the fitness of each genome to the training run value for training
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = fitnesses[i] * 10.0

        
        
def save_species_statistics(species, generation):
    """
    Save the species statistics in a csv file
    
    Args:
        species: (dict) Species dictionary
        generation: (int) Current generation
    """
    mode = 'w' if generation == 0 else 'a'
    
    with open(SAVING_FOLDER +'species.csv', mode, newline='') as file:
        writer = csv.writer(file)
        for specie_id, specie in iteritems(species):
            for agent_id, agent in iteritems(specie.members):
                writer.writerow([generation, specie_id, agent_id, agent.fitness])
        
        
def visualize(genome, graph_viz_path=None):
    """
    Receives a genome and draws a neural network with arbitrary topology.
    
    Args:
        genome: (neat.genome.DefaultGenome) Genome to draw
        graph_viz_path: (str) Path to the graphviz executable
    """
    
    # Check if the graphviz library is installed
    if graph_viz_path is not None:
        os.environ["PATH"] += os.pathsep + graph_viz_path
    # Try to import the graphviz library
    try:
        import graphviz
    except ImportError:
        warnings.warn('Graphviz library not found')
        return
    
    # Set the node names and colors
    node_names = {}
    node_colors = {}
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    # Create the graph
    dot = graphviz.Digraph(format='svg', node_attr=node_attrs)

    # Add the nodes and connections
    inputs = set()
    for k in neat.config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    # Add the output nodes
    outputs = set()
    for k in neat.config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    # Add the hidden nodes
    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        attrs = {'style': 'filled',
                'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    # Add the connections
    show_disabled = True
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            val_weight='%.3f'%(cg.weight)
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width},label=val_weight)

    # Save the graph
    try:
        dot.render('checkpoints/network_view', view=False)
    except Exception:
        warnings.warn('Graphviz library not found')


def load_map(MAP_FOLDER, index=0):
    
    """
    Load the map grid and coordinates of the spawn point
    
    args:
        MAP_FOLDER: (str) Path to the map folder, None for mixed maps during training
        
    return:
        grid: (array) Grid of the map
        PLAYER_POS: (tuple) Coordinates of the spawn point
    """
    
    if MAP_FOLDER is None:
        # pick a randdom map folder
        list_maps = os.listdir('maps/')
        MAP_FOLDER = 'maps/' + np.random.choice(list_maps) + '/'
        
    with open(MAP_FOLDER + 'spawn.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            PLAYER_POS = (float(row[0]), float(row[1]))
    
    grid = Grid(250, MAP_FOLDER + 'circuit.png')
    
    map_name = MAP_FOLDER.split('/')[-2]
    
    return grid, PLAYER_POS, map_name
        


if __name__ == '__main__':
    # Configuration
    GAME_GRAPHICS = False
    LOAD_CHECKPOINT = False
    BENCHMARK_PAILLON = True
    SAVE_BEST_GENOME_ONLY = False
    
    # File paths
    SAVING_FOLDER = 'checkpoints/test/'
    MAP_FOLDER = None #'maps/circuit_paillon/' # Path to the map folder, None for mixed maps during training
    PROB_CHANGE_MAP = 0.2
    CONFIG_FILE = 'brain/config.txt'
    CHECKPOINT_FILE = 'checkpoints/gen39-fit1.2614408462209652'
    GRAPH_VIZ_PATH = os.path.curdir + '/graphviz/bin/' # Path to the graphviz executable
    
    # Simulation parameters
    GENERATIONS = 100       # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum lifetime of a player simulation
    PLAYER_RAY_COUNT = 5    # Number of rays to cast from the player
    DT = 0.01               # Time step for the simulation
    
    # Load the circuit
    grid, PLAYER_POS, map_name = load_map(MAP_FOLDER)
    # Create the save folder
    os.makedirs(SAVING_FOLDER, exist_ok=True)

    if GAME_GRAPHICS:
        import pygame
        from game.game_graphics import GameGraphics
        
        if LOAD_CHECKPOINT:
            # Create and run the NEAT algorithm loading from a checkpoint
            neat = NeatAlgorithm(CONFIG_FILE, CHECKPOINT_FILE)
            
            # Create the network from the genome
            net = neat.create_network(neat.best)
            
            # Visualize the network
            visualize(neat.best, GRAPH_VIZ_PATH)
        
        while True:
            # Create the game with graphics
            game = GameGraphics(grid, PLAYER_POS)
            
            # Run the game for the player
            frame_time_store = [0.0] * 10
            frame_time_avg = 1.0
            last_frame_time = time.time()
            while not game.game_over:
                frame_time = time.time()
                
                # Update the game state
                if not LOAD_CHECKPOINT:
                    cur_time = time.time()
                    dt = cur_time - last_frame_time
                    last_frame_time = cur_time
                else:
                    dt = DT
                game.tick(dt)
                
                # Handle window events
                game.events()
                
                if not LOAD_CHECKPOINT:
                    # Get the inputs of the current game state
                    acc, steer = game.key_inputs()
                    
                    # Update the game state
                    game.update(acc, steer)
                else:
                    # Compute the iteration count based on the frame time, so that the game runs at DT
                    it_count = max(int(DT / frame_time_avg), 1)
                    
                    # Run the game for the player
                    for it in range(it_count):
                        # Get the inputs of the current game state
                        inputs = game.get_inputs(PLAYER_RAY_COUNT)
                        
                        # Get the outputs from the neat network
                        outputs = net.activate(inputs)
                        
                        # Execute the action on the game
                        acc, steer = map_outputs(outputs, game.dt, game.player)
                
                        # Update the game state
                        game.update(acc, steer)
                        
                        # Check if the game is over
                        if game.game_over:
                            break
                
                # Display the game
                game.draw()
                
                # Update the frame time
                frame_time_store.pop(0)
                frame_time_store.append(time.time() - frame_time)
                frame_time_avg = sum(frame_time_store) / 10
    
        # Quit the window
        pygame.quit()
    else:
        # Create and run the NEAT algorithm
        neat = NeatAlgorithm(CONFIG_FILE)
        neat.run(eval_genomes, GENERATIONS)
    
    