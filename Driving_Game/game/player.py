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
        self.acc_mult = 0.005
        self.steer_mult = 0.005
        self.brake_acc = 30.0
        
        # Vision related
        self.fov = np.deg2rad(45)
        self.view_distance = 0.2
        self.wall_dx = 0.01
        
        # Friction
        self.friction = 10.0
        
        # Player related
        self.distance = 0.0
        self.time = 0.0
        
        # Maximum values
        self.max_vel = 0.5
        self.max_acc = 1.2
        self.max_steer = 0.4
        
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
        max_steer = max(0, self.max_steer - np.abs(self.vel[0]) / self.max_vel * self.max_steer * 0.8)
        
        # Limit the acceleration and steering
        acc = np.clip(acc * self.acc_mult, -self.brake_acc, self.max_acc)
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
        
        # Update the distance
        self.distance += self.vel[0] * dt
        self.time += dt
        
    def get_distance(self):
        """
        Get the current distance traveled by the player
        """
        return self.distance
    
    def get_time(self):
        """
        Get the current time elapsed
        """
        return self.time
    
    def get_inputs(self, grid, ray_count):
        """
        Get the inputs for the NEAT network as a tuple (vel_x, vel_y, acc, steer, rot)
        
        Args:
            grid: the grid with the map
            ray_count: number of rays for the player's field of view
        """
        # Compute rays distance
        distances = np.zeros(ray_count)
        for i in range(ray_count):
            # Compute ray direction
            angle = self.rot + (i - ray_count // 2) * self.fov / ray_count
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Compute the distance to the nearest wall
            distance = 0
            while True:
                if distance + self.wall_dx > self.view_distance:
                    break
                distance += self.wall_dx
                ray_pos = self.pos + distance * ray_dir
                (grid_x, grid_y) = (np.ceil(ray_pos[0] * grid.grid.shape[0]), np.ceil(ray_pos[1] * grid.grid.shape[1]))
                if grid.grid[int(grid_x), int(grid_y)] == 1:
                    break
            distances[i] = distance
        
        # Return the inputs
        return (self.vel[0], self.vel[1], self.rot, *distances)
