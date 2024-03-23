import pygame
import time
from game.cameras import MapCamera, PlayerCamera
from game.display import Display
from game.game import Game
import numpy as np

class GameGraphics(Game):
    def __init__(self, grid):
        """
        Initialize a game instance with graphics
        
        Args:
            grid: the grid instance
        """
        # Initialize parent class
        super().__init__(grid, 0)
        
        # Render logic
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.last_dt = time.time()
        self.running = True
        self.last_camera_change = 0
        
        # Initialize components
        self.camera = PlayerCamera(self.grid, self.player)
        self.display = Display()
        self.acc = 0.0
        self.steer = 0.0
        
    def tick(self, dt=None):
        """
        Update the game state
        
        Parameters:
            dt: time step : default=None (use the internal clock)
        """
        # Update the clock
        if dt is None:
            dt_now = time.time()
            self.dt = dt_now - self.last_dt
            self.last_dt = dt_now
        else:
            self.dt = dt
        
    def key_inputs(self):
        """
        Get the keyboard inputs for the player
        
        Returns:
            acc: acceleration value
            steer: steering value
        """
        return self.acc, self.steer
        
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
        if keys[pygame.K_z]:
            self.acc += 1.0
        elif keys[pygame.K_s]:
            self.acc = -self.player.brake_acc
        else:
            self.acc = 0.0
        
        # Handle keyboard inputs for steering
        if keys[pygame.K_d]:
            if np.sign(self.steer) == -1:
                self.steer = 0.0
            self.steer += 1.0
        if keys[pygame.K_q]:
            if np.sign(self.steer) == 1:
                self.steer = 0.0
            self.steer -= 1.0
        if not keys[pygame.K_d] and not keys[pygame.K_q]:
            self.steer = 0.0
                
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
        self.display.display(self.screen, "LEFT", format("Player infos:".format(*self.player.pos)))
        self.display.display(self.screen, "LEFT", format(" - Position: ({:.2f}, {:.2f})".format(*self.player.pos)))
        self.display.display(self.screen, "LEFT", format(" - Velocity: ({:.2f}, {:.2f})".format(*self.player.vel)))
        self.display.display(self.screen, "LEFT", format(" - Rotation: {:.2f}".format(self.player.rot)))
        self.display.display(self.screen, "LEFT", format(" - Acceleration: {:.2f}".format(np.clip(self.acc, -self.player.max_acc, self.player.max_acc))))
        self.display.display(self.screen, "LEFT", format(" - Steering: {:.2f}".format(np.clip(self.steer, -self.player.max_steer, self.player.max_steer))))
        self.display.display(self.screen, "LEFT", format(" - Distance: {:.2f}".format(self.player.distance)))
        self.display.display(self.screen, "LEFT", format("Camera: {}".format(camera_name)))
        
        self.display.display(self.screen, "RIGHT", format("Z/S: Accelerate/Brake"))
        self.display.display(self.screen, "RIGHT", format("Q/D: Turn Left/Right"))
        self.display.display(self.screen, "RIGHT", format("P/M: Zoom In/Out"))
        self.display.display(self.screen, "RIGHT", format("C: Change Camera"))
        
        # Swap the surface buffers
        pygame.display.flip()
