import numpy as np
import matplotlib.pyplot as plt
import neat
import os

def hist_final_fitness_parameter(title):
	"""
	Plot the histogram of the final fitness of the different runs of the same experiment, varying a parameter.

	:param title: the title of the experiment (the name of the folder in 'checkpoints')
	:return: None
	"""
    
    # read all subfolders in the directory 'checkpoints\folder'
	main_folder = 'checkpoints\\' + title +'\\'
	subfolders = os.listdir(main_folder)

	fits_max = []
	for i, subfolder in enumerate(subfolders):
		# read all filenames in the directory 'checkpoints\folder'
		filenames = os.listdir(main_folder + subfolder)

		fit_max = 0
		for filename in filenames:
			# ignore the configuration file
			if filename == 'config.txt':
				continue

			# extract the fitness value (the file name is like "gen8-fit0.203-paillon")
			fit = float(filename.split('-')[1][3:])
			# update the maximum fitness
			fit_max = max(fit_max, fit)

		# store the maximum fitness of the run
		fits_max.append(fit_max)
	
	plt.bar(np.arange(len(fits_max)), fits_max)
	fig = plt.gcf()
	fig.set_size_inches(14, 8)
	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	plt.xlabel('Run', fontsize=40)
	plt.ylabel('Fitness', fontsize=40)
	plt.tight_layout()

def my_color_func(x):
	"""
	Define a color function to plot the different runs of the same experiment.
		
	:param x: the species number
	:return: a color
	"""
	x = x % 6
	if x == 0: # green
		return (106/255, 168/255, 79/255)
	if x == 1:	# blue
		return (60/255, 120/255, 216/255)
	if x == 2: # purple
		return (153/255, 51/255, 255/255)
	if x == 3: # red
		return (214/255, 10/255, 10/255)
	if x == 4: # orange
		return (240/255, 133/255, 40/255)
	if x == 5:
		return (128/255, 128/255, 128/255)
	else:
		return (255/255, 255/255, 255/255)

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

	for s in np.unique(species):
		# get the indices of each species
		inds1 = np.where(species == s)[0]
		gs, max_fits = [], []

		for g in np.unique(gens):
			# get the indices of each generation
			inds2 = np.where(gens == g)[0]
			inds = np.intersect1d(inds1, inds2)
			if len(inds) == 0:
				continue

			# store the generation number and the maximum fitness of the species
			gs.append(g)
			max_fits.append(np.max(fitness[inds]))

		# sort the lists by generation number
		inds = np.argsort(gs)
		gs = np.array(gs)[inds]
		max_fits = np.array(max_fits)[inds]
  
		# do a light running average to smooth the curves
		try:
			max_fits = np.convolve(max_fits, np.ones(3) / 3, mode='valid')
			gs = gs[1:-1]
			plt.plot(gs, max_fits, label='Species ' + str(int(s)), alpha=0.8, linewidth=5, color=my_color_func(int(s - 1)))
		except:
			continue

	plt.xlabel('Generation', fontsize=40)
	plt.ylabel('Fitness', fontsize=40)
	plt.yscale('log')
	plt.legend(fontsize=35)
	fig = plt.gcf()
	fig.set_size_inches(14*2, 8*2)
	plt.xticks(fontsize=30)
	plt.yticks(fontsize=30)
	plt.tight_layout()

def plot_best_fitness_from_csv(title, color='k', linestyle='--', label=None):
	"""
	Plot the best fitness of a single run.

    Args:
		title: the title of the run (the name of the folder in 'checkpoints')
		color: the color of the line
		linestyle: the linestyle of the line
		label: the label of the run
	"""
	
	main_folder = 'checkpoints\\' + title +'\\'

	# load the csv file
	data = np.loadtxt(main_folder + 'species.csv', delimiter=',')
	gens = data[:, 0]
	species = data[:, 1]
	agents = data[:, 2]
	fitness = data[:, 3]


	max_fit = []
	for g in np.unique(gens):
		# get the indices of each generation
		inds = np.where(gens == g)[0]
		if len(inds) == 0:
			continue

		# store the generation number and the maximum fitness of the species for each generation
		max_fit.append(np.max(fitness[inds]))

	# do a running average to smooth the curves
	max_fit = np.convolve(max_fit, np.ones(11) / 11, mode='valid')
	gs = np.unique(gens)[5:-5]
 
	plt.plot(gs, max_fit, label=label, alpha=0.8, linewidth=5, color=color, linestyle=linestyle)
	plt.yscale('log')
	plt.legend(fontsize=35)
	fig = plt.gcf()
	fig.set_size_inches(14*1.5, 8*1.5)
	plt.xticks(fontsize=35)
	plt.yticks(fontsize=35)
	plt.xlabel('Generation', fontsize=40)
	plt.ylabel('Fitness', fontsize=40)
	plt.tight_layout()