import numpy as np
from PIL import Image

class Grid:
    def __init__(self, grid_size, circuit_file):
        """
        Initialize the grid with a given size and circuit file
        Args:
            grid_size: Number of cells in the grid
            circuit_file: File containing the circuit (image file)
        """
        self.GRID_SIZE = grid_size
        self.grid = np.zeros((grid_size, grid_size)) # 0: empty, 1: wall
        
        # Open and read the circuit file pixels
        im = Image.open(circuit_file)
        pixels = im.load()
        
        # Load the circuit
        for i in range(grid_size):
            for j in range(grid_size):
                # Average the RGB values in this cell
                r, g, b = 0, 0, 0
                for x in range(i * im.width // grid_size, (i + 1) * im.width // grid_size):
                    for y in range(j * im.height // grid_size, (j + 1) * im.height // grid_size):
                        (r_, g_, b_) = pixels[x, y]
                        r += r_
                        g += g_
                        b += b_
                r, g, b = r // (im.width // grid_size) ** 2, g // (im.width // grid_size) ** 2, b // (im.width // grid_size) ** 2
                
                # If the cell is mostly white, it is a wall
                if r + g + b >= 255:
                    self.grid[i, j] = 1
                    