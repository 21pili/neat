import numpy as np

class Player:
    def __init__(self, pos):
        self.pos = np.array([pos[0], pos[1]]) # x, y in [0, 1]
        self.vel = np.array([0.0, 0.0])
        self.rot = 0
        self.ang = 0
        
        # Car dimensions
        self.width = 0.01
        self.height = 0.02
        
    def update(self, dt, dvel, dang):
        # Update the position and rotation
        self.pos += self.vel * dt
        self.rot += self.ang * dt
        
        # Update the velocity and angular velocity
        self.vel += dvel
        self.ang += dang
        
        # Add some friction
        self.vel *= 0.99
        self.ang *= 0.99
            
        # Cap the speeds
        MAX_SPEED = 0.5
        MAX_ANG = 2.0
        if np.linalg.norm(self.vel) > MAX_SPEED:
            self.vel = self.vel / np.linalg.norm(self.vel) * MAX_SPEED
        if np.abs(self.ang) > MAX_ANG:
            self.ang = np.sign(self.ang) * MAX_ANG