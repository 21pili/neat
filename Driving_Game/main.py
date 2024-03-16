import pygame
import numpy as np
from game.game import Game, GameGraphics
from game.grid import Grid

# # Run the game with graphics and keyboard inputs
# if __name__ == "__main__":
#     # Create the game
#     grid = Grid(250, 'circuit.png')
#     game = GameGraphics(grid)

#     # Run the game
#     while not game.game_over:
#         game.update()
    
#     # Wait for 3 seconds before quitting
#     pygame.time.wait(3000)
    
#     # Quit the window
#     pygame.quit()


# Run the game with the NEAT algorithm
if __name__ == "__main__":
    # Load the circuit
    grid = Grid(250, 'circuit.png')
    
    # Algorithm parameters
    GENERATIONS = 10 # Total number of generations
    PLAYER_COUNT = 10 # Number of players per generation
    PLAYER_MAX_TIME = 100.0 # Maximum time for a player to run
    
    # Run algorithm
    for gen in range(GENERATIONS):
        print(f"Simulating generation {gen + 1}/{GENERATIONS}")
        
        # Create the players for this generation # TODO:
        players = [Game(grid) for _ in range(PLAYER_COUNT)]
        
        # Run the game for each player
        for game in players:
            elapsed_time = 0.0
            while not game.game_over and elapsed_time < PLAYER_MAX_TIME:
                # Get the inputs to the neat network
                inputs = game.get_inputs()
                
                # Predict the next action using neat
                action = (0.1, 0.0) # TODO:
                
                # Execute the action
                game.update(action[0], action[1])
                
                # Update the elapsed time
                elapsed_time += game.dt
                
        # Compute fitness for the players
        fitness = [0.0 for _ in range(PLAYER_COUNT)]
        for i, game in enumerate(players):
            fitness_params = game.get_fitness_parameters()
            fitness[i] = 0.0 # TODO:
        
        # Mutations, speciation, crossover, etc.
        # TODO:
    