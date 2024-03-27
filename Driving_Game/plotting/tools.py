import numpy as np
import matplotlib.pyplot as plt
import neat
import os

def plot_fitness_evolution_parameter(title):
	"""
	Plot the fitness evolution of the different runs of the same experiment.

	:param title: the title of the experiment (the name of the folder in 'checkpoints')
	:return: None
	"""
    
	main_folder = 'checkpoints\\' + title +'\\'
	subfolders = os.listdir(main_folder)

	for i, subfolder in enumerate(subfolders):
		# read all filenames in the directory 'checkpoints\folder'
		filenames = os.listdir(main_folder + subfolder)

		fits, gens = [], []

		for filename in filenames:
			# ignore the config file
			if filename == 'config.txt':
				continue

			# extract the generation number and the fitness value (the file name is like "gen8-fit0.203-paillon")
			gen = int(filename.split('-')[0][3:])
			fit = float(filename.split('-')[1][3:])
			gens.append(gen)
			fits.append(fit)

		# sort the lists by generation number
		inds = np.argsort(gens)
		gens = np.array(gens)[inds]
		fits = np.array(fits)[inds]

		plt.plot(gens, fits, label=subfolder, alpha=0.8, linewidth=2, color=my_color_func(i))

	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend()
	plt.title(title)
	plt.show()

def my_color_func(x):
	"""
	Define a color function for the plot_fitness_evolution_species function.

	:param x: the species number
	:return: a color
	"""
	x = x % 5
	return plt.cm.viridis(x / 5)

def plot_fitness_diff_species_from_csv(title):
	"""
	Plot the fitness evolution of the different species of the same experiment.

    Args:
        title: the title of the run (the name of the folder in 'checkpoints')
	:return: None
	"""
	
	main_folder = 'checkpoints\\' + title +'\\'

	# load the csv file
	data = np.loadtxt(main_folder + 'species.csv', delimiter=',')
	gens = data[:, 0]
	species = data[:, 1]
	agents = data[:, 2]
	fitness = data[:, 3]
 
	# plot the fitness of each species
	for s in np.unique(species):
		inds1 = np.where(species == s)[0]
		gs, max_fits = [], []
		for g in np.unique(gens):
			inds2 = np.where(gens == g)[0]
			inds = np.intersect1d(inds1, inds2)
			if len(inds) == 0:
				continue
			gs.append(g)
			max_fits.append(np.max(fitness[inds]))
   
		# sort the lists by generation number
		inds = np.argsort(gs)
		gs = np.array(gs)[inds]
		max_fits = np.array(max_fits)[inds]
		plt.plot(gs, max_fits, label='specie ' + str(int(s)), alpha=0.8, linewidth=2, color=my_color_func(int(s)))

	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend()
	plt.title(title)

plot_fitness_diff_species_from_csv('test')
plt.show()