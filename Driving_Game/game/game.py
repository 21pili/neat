import numpy as np
from game.player import Player

class Game:
    def __init__(self, grid, dt=0.01):
        """
        Initialize a game instance without graphics
        
        Args:
            grid: the grid instance
            dt: time step
        """
        # Game current status
        self.game_over = False
        self.dt = dt
        
        # Initialize game state
        self.grid = grid
        self.player = Player((0.5, 0.5))
        
    def update(self, acc, steer):
        """
        Tick for the game instance
        
        Args:
            acc: Acceleration to apply
            steer: Steering angle to apply
        """
        # Store the acc and steer values
        self.acc = acc
        self.steer = steer
        
        # Update the player position
        self.player.update(self.dt, acc, steer)
            
        # Check for circuit collisions
        forward = np.array([np.cos(self.player.rot), np.sin(self.player.rot)])
        left = np.array([np.cos(self.player.rot + np.pi / 2), np.sin(self.player.rot + np.pi / 2)])
        pos = self.player.pos + np.array([self.player.width, self.player.height / 2])
        (p_w, p_h) = (self.player.width, self.player.height)
        p1 = (pos[0] + forward[0] * p_h / 2 + left[0] * p_w / 2, pos[1] + forward[1] * p_h / 2 + left[1] * p_w / 2)
        p1_cell = (int(p1[0] * self.grid.GRID_SIZE), int(p1[1] * self.grid.GRID_SIZE))
        p2 = (pos[0] + forward[0] * p_h / 2 - left[0] * p_w / 2, pos[1] + forward[1] * p_h / 2 - left[1] * p_w / 2)
        p2_cell = (int(p2[0] * self.grid.GRID_SIZE), int(p2[1] * self.grid.GRID_SIZE))
        p3 = (pos[0] - forward[0] * p_h / 2 - left[0] * p_w / 2, pos[1] - forward[1] * p_h / 2 - left[1] * p_w / 2)
        p3_cell = (int(p3[0] * self.grid.GRID_SIZE), int(p3[1] * self.grid.GRID_SIZE))
        p4 = (pos[0] - forward[0] * p_h / 2 + left[0] * p_w / 2, pos[1] - forward[1] * p_h / 2 + left[1] * p_w / 2)
        p4_cell = (int(p4[0] * self.grid.GRID_SIZE), int(p4[1] * self.grid.GRID_SIZE))
        
        # Find max and min cells
        min_x = min(p1_cell[0], p2_cell[0], p3_cell[0], p4_cell[0])
        max_x = max(p1_cell[0], p2_cell[0], p3_cell[0], p4_cell[0])
        min_y = min(p1_cell[1], p2_cell[1], p3_cell[1], p4_cell[1])
        max_y = max(p1_cell[1], p2_cell[1], p3_cell[1], p4_cell[1])
        
        # Check for collisions in the bounding box
        rw = 1 / self.grid.GRID_SIZE
        rh = 1 / self.grid.GRID_SIZE
        self.grid.red_cells = []
        for x in range(min_x, max_x + 1):
            rx = x / self.grid.GRID_SIZE
            for y in range(min_y, max_y + 1):
                if x >= 0 and x < self.grid.grid.shape[0] and y >= 0 and y < self.grid.grid.shape[1] and self.grid.grid[x, y] == 1:
                    ry = y / self.grid.GRID_SIZE
                    if Game.line_rect_collision(p1[0], p1[1], p2[0], p2[1], rx, ry, rw, rh) or \
                       Game.line_rect_collision(p2[0], p2[1], p3[0], p3[1], rx, ry, rw, rh) or \
                       Game.line_rect_collision(p3[0], p3[1], p4[0], p4[1], rx, ry, rw, rh) or \
                       Game.line_rect_collision(p4[0], p4[1], p1[0], p1[1], rx, ry, rw, rh):
                        self.game_over = True
                        self.grid.red_cells.append((x, y))
                        break
        
        
    
    def line_line_collision(x1, y1, x2, y2, x3, y3, x4, y4):
        uA = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
        uB = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
        if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
            return True
        return False
    
    def line_rect_collision(x1, y1, x2, y2, rx, ry, rw, rh):
        left   = Game.line_line_collision(x1,y1,x2,y2, rx,ry,rx, ry+rh)
        right  = Game.line_line_collision(x1,y1,x2,y2, rx+rw,ry, rx+rw,ry+rh)
        top    = Game.line_line_collision(x1,y1,x2,y2, rx,ry, rx+rw,ry)
        bottom = Game.line_line_collision(x1,y1,x2,y2, rx,ry+rh, rx+rw,ry+rh)
        if left or right or top or bottom:
            return True
        return False
    
    def get_inputs(self, ray_count):
        """
        Get the inputs for the NEAT network as a tuple (vel_x, vel_y, acc, steer, rot, ray_distance_1, ..., ray_distance_n)
        
        Args:
            ray_count: number of rays to cast (uniformly distributed in the player's field of view)
        """
        return self.player.get_inputs(self.grid, ray_count)
    
    def get_fitness_parameters(self):
        """
        Get the fitness parameters for the NEAT algorithm (distance, time)
        """
        return self.player.get_distance(), self.player.get_time()
