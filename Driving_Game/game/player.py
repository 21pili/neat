import numpy as np

class Player:
    def __init__(self, pos):
        """
        Initialize the player at a given position
        Args:
            pos: initial position (x, y)
        """
        self.pos = np.array([pos[0], pos[1]]) # x, y in [0, 1]
        self.vel = np.array([0.0, 0.0])
        self.acc = 0.0
        self.steer = 0.0
        self.rot = 0
        
        # Physical car performance
        self.acc_mult = 0.04
        self.steer_mult = 0.8
        self.brake_acc = 55.0
        
        # Friction
        self.friction = 10.0
        
        # Maximum values
        self.max_vel = 0.17
        self.max_acc = 1.2
        self.max_steer = np.deg2rad(40) # Around 40 degrees for usual cars
        
        # Car dimensions
        self.width = 0.01
        self.height = 0.02
        
    def update(self, dt, acc, steer):
        """
        Update the player's position and velocity
        
        Args:
            dt: time step
            acc: acceleration (float)
            steer: steering angle (float)
        """
        # Compute the maximum steering angle based on the velocity
        max_steer = self.max_steer - np.abs(self.vel[0]) / self.max_vel * self.max_steer * 0.8
        
        # Limit the acceleration and steering
        acc = np.clip(acc * self.acc_mult, -self.max_acc, self.max_acc)
        steer = np.clip(steer * self.steer_mult, -max_steer, max_steer)
        
        # Add a bit of friction
        if acc == 0:
            acc -= self.vel[0] * self.friction * dt
        
        # Update the linear velocity
        self.vel += np.array([acc * dt, 0.0]) # Car never accelerate sideways
        
        # Limit the velocity
        self.vel[0] = np.clip(self.vel[0], 0, self.max_vel)
        
        # If the car is turning, compute the angular velocity
        if steer != 0:
            # Rotate based on front wheel (offset from car center, so use Pythagoras th)
            turning_radius = self.height / np.sin(steer)
            angular_vel = self.vel[0] / turning_radius
        else:
            angular_vel = 0
            
        # Update the position and rotation
        self.pos += np.array([self.vel[0] * np.cos(self.rot) * dt, self.vel[0] * np.sin(self.rot) * dt])
        self.rot += angular_vel * dt
        