import pygame
import numpy as np
import time
from game.cameras import MapCamera, PlayerCamera
from game.display import Display
from game.grid import Grid
from game.player import Player

class Game:
    def __init__(self, circuit_file, dt):
        """
        Initialize a game instance without graphics
        """
        # Game current status
        self.game_over = False
        self.dt = dt
        
        # Initialize game state
        self.grid = Grid(250, circuit_file)
        self.player = Player((0.5, 0.5))
        
    def update(self, acc, steer):
        """
        Tick for the game instance
        
        Args:
            acc: Acceleration to apply
            steer: Steering angle to apply
        """
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
            

class GameGraphics(Game):
    def __init__(self, circuit_file):
        """
        Initialize a game instance with graphics
        """
        # Initialize parent class
        super().__init__(circuit_file, 0)
        
        # Render logic
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.last_input = time.time()
        self.running = True
        self.last_camera_change = 0
        
        # Initialize components
        self.camera = PlayerCamera(self.grid, self.player)
        self.display = Display()
        self.acc = 0.0
        self.steer = 0.0
        
    def update(self):
        """
        Tick for the game instance
        """
        # Update the game state based on the player inputs
        super().update(self.acc, self.steer)
        
        # Tick for the display loop
        self.events()
        self.draw()
        
        # Update the clock
        self.dt = self.clock.tick(60) / 1000.0
        
    def events(self):
        """
        Handle window events (close, keyboard inputs)
        """
        # Poll for window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        # Handle camera changes
        keys = pygame.key.get_pressed()
        if keys[pygame.K_c] and pygame.time.get_ticks() - self.last_camera_change > 200:
            if isinstance(self.camera, MapCamera):
                self.camera = PlayerCamera(self.grid, self.player)
            else:
                self.camera = MapCamera(self.grid, self.player)
            self.last_camera_change = pygame.time.get_ticks()
            
        # Handle keyboard inputs for acceleration
        time_now = time.time()
        dt_input = time_now - self.last_input
        if keys[pygame.K_z]:
            self.acc += 1.0 * dt_input
        elif keys[pygame.K_s]:
            self.acc = -self.player.brake_acc * dt_input
        else:
            self.acc = 0.0
        
        # Handle keyboard inputs for steering
        if keys[pygame.K_d]:
            self.steer += 1.0 * dt_input
        elif keys[pygame.K_q]:
            self.steer -= 1.0 * dt_input
        else:
            self.steer = 0.0
        self.last_input = time_now
                
        # Handle keyboard inputs for the camera
        self.camera.input(self.dt)

    def draw(self):
        """
        Draw the game state
        """
        # Blank the screen
        self.screen.fill("white")
        
        # Draw the view
        self.camera.draw(self.screen)
        
        # Display debug information
        self.display.clear()
        camera_name = "Map" if isinstance(self.camera, MapCamera) else "Player"
        self.display.display(self.screen, "LEFT", format("FPS: {:.2f}".format(self.clock.get_fps())))
        self.display.display(self.screen, "LEFT", format("Player infos:".format(*self.player.pos)))
        self.display.display(self.screen, "LEFT", format(" - Position: ({:.2f}, {:.2f})".format(*self.player.pos)))
        self.display.display(self.screen, "LEFT", format(" - Velocity: ({:.2f}, {:.2f})".format(*self.player.vel)))
        self.display.display(self.screen, "LEFT", format(" - Rotation: {:.2f}".format(self.player.rot)))
        self.display.display(self.screen, "LEFT", format(" - Acceleration: {:.2f}".format(np.clip(self.acc, -self.player.max_acc, self.player.max_acc))))
        self.display.display(self.screen, "LEFT", format(" - Steering: {:.2f}".format(np.clip(self.steer, -self.player.max_steer, self.player.max_steer))))
        self.display.display(self.screen, "LEFT", format("Camera: {}".format(camera_name)))
        
        self.display.display(self.screen, "RIGHT", format("Z/S: Accelerate/Brake"))
        self.display.display(self.screen, "RIGHT", format("Q/D: Turn Left/Right"))
        self.display.display(self.screen, "RIGHT", format("P/M: Zoom In/Out"))
        self.display.display(self.screen, "RIGHT", format("C: Change Camera"))
        
        # Swap the surface buffers
        pygame.display.flip()
