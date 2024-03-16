import pygame
import numpy as np
from game.game import Game, GameGraphics


if __name__ == "__main__":
    # Run the game
    game = GameGraphics('circuit.png')
    # game = Game('circuit.png', dt=0.01)
    while not game.game_over:
        game.update()
        # game.update(np.array([0.01, 0.0]), 0.0)
    print("Game over")
        
    # Wait for 3 seconds
    pygame.time.wait(3000)
        
    # Quit the game
    pygame.quit()
    