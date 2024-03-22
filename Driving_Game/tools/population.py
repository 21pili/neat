import neat

# Create the population and add reporters to show progress in the terminal
def create_population(config):
	pop = neat.Population(config)
	pop.add_reporter(neat.StdOutReporter(True))
	pop.add_reporter(neat.StatisticsReporter())
	# pop.add_reporter(neat.Checkpointer(5))
	return pop