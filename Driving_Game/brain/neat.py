import os
import neat

class NeatAlgorithm:
    """
    Class to run the NEAT algorithm with the given configuration file using the
    NEAT-Python library. The class is used to run the algorithm for a given number
    of generations and evaluate the fitness of the genomes using a fitness function.
    """
    def __init__(self, config_file):
        """
        Initialize the NEAT algorithm with the configuration file
        
        Args:
            config_file: relative path to the configuration file
        """
        # Load the configuration file
        local_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            os.path.join(local_dir, config_file))
        
        # Create the NEAT population and set the reporters
        self.population = neat.Population(self.config)
        self.population.add_reporter(neat.StdOutReporter(True)) # Set to False to disable output
        self.population.add_reporter(neat.StatisticsReporter()) # Enable statistics
        self.population.add_reporter(neat.Checkpointer(1, filename_prefix='checkpoints/checkpoint-')) # Save the population every 5 generations
        
        
    def run(self, fitness_function, generations):
        """
        Run the NEAT algorithm for a given number of generations.
        The fitness function is used to evaluate the genomes and will be called
        for each genome in the population, for each generation. It should update
        the fitness of the genomes, without returning anything.
        
        Args:
            fitness_function: function to evaluate the genomes
            generations: number of generations to run
        """
        self.population.run(fitness_function, generations)
        
        
    def create_network(self, genome):
        """
        Create a feed-forward network from the given genome
        
        Args:
            genome: genome to create the network
        """
        return neat.nn.FeedForwardNetwork.create(genome, self.config)
        