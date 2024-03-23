import os
import time
import numpy as np
from brain.neat import NeatAlgorithm
from game.game import Game
from game.grid import Grid
from multiprocessing import Pool
import warnings


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
    # Set the current neat configuration
    neat.config = current_config
    
    # Run each genome
    with Pool() as pool:
        # Create the game instances and the networks
        games = [Game(grid, PLAYER_POS, DT) for _ in range(len(genomes))]
        neat_networks = [neat.create_network(genome) for _, genome in genomes]
        # Evaluate the fitness of each genome
        args = zip(games, zip(neat_networks, [(PLAYER_MAX_TIME, PLAYER_RAY_COUNT)] * len(genomes)))
        fitnesses = pool.map(run_simulation, args)
        
        # Set the fitness of each genome
        for i, (_, genome) in enumerate(genomes):
            genome.fitness = fitnesses[i] * 10.0
            
        # Save best genomes
        os.makedirs('checkpoints/', exist_ok=True)
        best = np.argmax(fitnesses)
        neat.save_genome('checkpoints/gen{}-fit{}'.format(neat.population.generation, fitnesses[best]), genomes[best][1])
        
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

if __name__ == '__main__':
    # Configuration
    GAME_GRAPHICS = False
    LOAD_CHECKPOINT = False
    
    # File paths
    CONFIG_FILE = 'brain/config.txt'
    CHECKPOINT_FILE = 'checkpoints/gen105-fit0.3985199999997709'
    GRAPH_VIZ_PATH = os.path.curdir + '/graphviz/bin/' # Path to the graphviz executable
    
    # Simulation parameters
    GENERATIONS = 100       # Total number of generations
    PLAYER_MAX_TIME = 100.0 # Maximum lifetime of a player simulation
    PLAYER_RAY_COUNT = 5    # Number of rays to cast from the player
    DT = 0.01               # Time step for the simulation
    
    # Load the yoshi circuit
    # PLAYER_POS = (0.5, 0.208)
    # grid = Grid(250, 'circuit_yoshi.png')
    
    # Load the paillon circuit
    PLAYER_POS = (0.468, 0.21)
    grid = Grid(250, 'circuit_paillon.png')

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
    
    