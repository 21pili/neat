import pygame
import numpy as np
from game.game import Game, GameGraphics

# # Run the game with graphics and keyboard inputs
# if __name__ == "__main__":
#     game = GameGraphics('circuit.png')
#     while not game.game_over:
#         game.update()
    
#     # Wait for 3 seconds before quitting
#     pygame.time.wait(3000)
    
#     # Quit the window
#     pygame.quit()


# Run the game with the NEAT algorithm
if __name__ == "__main__":
    # Create the game
    game = Game('circuit.png', dt=0.01)
    MAX_TIME = 100.0 # seconds
    
    # Create the NEAT network
    # TODO:
    
    # Run the game
    elapsed_time = 0.0
    while not game.game_over and elapsed_time < MAX_TIME:
        # Get the inputs to the neat network
        inputs = game.player.get_inputs()
        
        # Predict the next action using neat
        action = (0.1, 0.0) # TODO:
        
        # Execute the action
        game.update(action[0], action[1])
        
        # Compute reward
        dist = game.player.get_distance()
        time = game.player.get_time()
        reward = 0.0 # TODO:
        
        # Update the neat network
        # TODO:
        
        # Update the elapsed time
        elapsed_time += game.dt
    