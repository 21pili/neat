import pygame
import numpy as np

class MapCamera:
    """
    Camera that shows the entire grid.
    """
    def __init__(self, grid, player):
        """
        Args:
            grid: the grid to display
            player: the player
        """
        self.grid = grid
        self.player = player
        self.ZOOM = 1.0
        
    def draw(self, screen):
        """
        Draw the grid and the player
        """
        # Get view size
        view_size = (screen.get_width(), screen.get_height())
        view_range = min(view_size) * self.ZOOM
        
        # Draw grid
        off = 0.01
        for x in range(self.grid.GRID_SIZE):
            for y in range(self.grid.GRID_SIZE):
                if self.grid.grid[x, y] == 0:
                    x0 = x / self.grid.GRID_SIZE
                    y0 = y / self.grid.GRID_SIZE
                    x1 = (x + 1) / self.grid.GRID_SIZE
                    y1 = (y + 1) / self.grid.GRID_SIZE
                    pygame.draw.rect(screen, "black", (
                        MapCamera._xy_to_screen(view_size, (x0 - off, y0 - off), view_range),
                        MapCamera._xy_to_screen(view_size, ((x1 - x0) + off, (y1 - y0) + off), view_range, absolute=True)
                    ))
                    
        # Draw max range of the grid
        grid_min = MapCamera._xy_to_screen(view_size, (0, 0), view_range)
        grid_max = MapCamera._xy_to_screen(view_size, (1, 1), view_range)
        pygame.draw.rect(screen, "red", (grid_min, (grid_max[0] - grid_min[0], grid_max[1] - grid_min[1])), 1)
        
        # Draw the player at center
        forward = np.array([np.cos(self.player.rot), np.sin(self.player.rot)])
        left = np.array([np.cos(self.player.rot + np.pi / 2), np.sin(self.player.rot + np.pi / 2)])
        (p_w, p_h) = MapCamera._xy_to_screen(view_size, (self.player.width, self.player.height), view_range, absolute=True)
        player_pos = MapCamera._xy_to_screen(view_size, self.player.pos, view_range)
        pygame.draw.polygon(screen, "red", [
            (player_pos[0] + forward[0] * p_h / 2 + left[0] * p_w / 2, player_pos[1] + forward[1] * p_h / 2 + left[1] * p_w / 2),
            (player_pos[0] + forward[0] * p_h / 2 - left[0] * p_w / 2, player_pos[1] + forward[1] * p_h / 2 - left[1] * p_w / 2),
            (player_pos[0] - forward[0] * p_h / 2 - left[0] * p_w / 2, player_pos[1] - forward[1] * p_h / 2 - left[1] * p_w / 2),
            (player_pos[0] - forward[0] * p_h / 2 + left[0] * p_w / 2, player_pos[1] - forward[1] * p_h / 2 + left[1] * p_w / 2)
        ])
        pygame.draw.line(screen, "black", player_pos, (player_pos[0] + forward[0] * p_h / 2, player_pos[1] + forward[1] * p_h / 2), 4)
        
    def input(self, dt):
        """
        Handle keyboard inputs
        """
        ZOOM_SPEED = 0.5
        keys = pygame.key.get_pressed()
        if keys[pygame.K_p]:
            self.ZOOM += ZOOM_SPEED * dt
        if keys[pygame.K_m]:
            self.ZOOM -= ZOOM_SPEED * dt
        
    def _xy_to_screen(view_size, pos, view_range, absolute=False):
        """
        Convert a position in the grid to a position on the screen.
        
        Args:
            view_size: the size of the screen
            pos: the position in the grid (x, y) in [0, 1]
            view_range: the range of the view
            player_pos: the position of the player
            absolute: if True, the position is given in absolute coordinates
            
        Returns:
            the position on the screen
        """
        if not absolute:
            return (
                pos[0] * view_range + (view_size[0] - view_range) / 2,
                pos[1] * view_range + (view_size[1] - view_range) / 2
            )
        return (
            pos[0] * view_range,
            pos[1] * view_range
        )


class PlayerCamera:
    """
    Camera that follows the player and shows the grid around it.
    """
    def __init__(self, grid, player):
        """
        Args:
            grid: the grid to display
            player: the player
        """
        self.grid = grid
        self.player = player
        self.ZOOM = 10.0
        self.player_size = 0.05
        
    def draw(self, screen):
        """
        Draw the grid and the player
        """
        # Get view size
        view_size = (screen.get_width(), screen.get_height())
        
        # Compute the view extent (implement basic clipping for the grid)
        view_range = self.ZOOM
        screen_min = PlayerCamera._screen_to_xy(view_size, (-view_size[0]*0.2, -view_size[1]*0.2), view_range, self.player.pos)
        screen_max = PlayerCamera._screen_to_xy(view_size, (view_size[0]*1.2, view_size[1]*1.2), view_range, self.player.pos)
        extend = (max(0, screen_min[0]), min(1, screen_max[0]), max(0, screen_min[1]), min(1, screen_max[1]))
        
        # Draw grid
        off = 0.01
        grid_min_i = (int(extend[0] * self.grid.GRID_SIZE), int(extend[2] * self.grid.GRID_SIZE))
        grid_max_i = (int(extend[1] * self.grid.GRID_SIZE), int(extend[3] * self.grid.GRID_SIZE))
        for i in range(grid_min_i[0], grid_max_i[0]):
            for j in range(grid_min_i[1], grid_max_i[1]):
                if i >= 0 and i < self.grid.GRID_SIZE and j >= 0 and j < self.grid.GRID_SIZE:
                    is_in_red_cells = (i, j) in self.grid.red_cells
                    color = "blue" if is_in_red_cells else "black"
                    if self.grid.grid[i, j] != 0 and not is_in_red_cells:
                        color = "white"
                    x0 = i / self.grid.GRID_SIZE
                    y0 = j / self.grid.GRID_SIZE
                    x1 = (i + 1) / self.grid.GRID_SIZE
                    y1 = (j + 1) / self.grid.GRID_SIZE
                    
                    pygame.draw.rect(screen, color, (
                        PlayerCamera._xy_to_screen(view_size, (x0 - off, y0 - off), view_range, self.player.pos),
                        PlayerCamera._xy_to_screen(view_size, (x1 - x0 + off, y1 - y0 + off), view_range, self.player.pos, absolute=True)
                    ))
                    
        # Draw max range of the grid
        grid_min = PlayerCamera._xy_to_screen(view_size, (0, 0), view_range, self.player.pos)
        grid_max = PlayerCamera._xy_to_screen(view_size, (1, 1), view_range, self.player.pos)
        pygame.draw.rect(screen, "red", (grid_min, (grid_max[0] - grid_min[0] + 2, grid_max[1] - grid_min[1] + 2)), 1)
        
        # Draw the player at center
        forward = np.array([np.cos(self.player.rot), np.sin(self.player.rot)])
        left = np.array([np.cos(self.player.rot + np.pi / 2), np.sin(self.player.rot + np.pi / 2)])
        (p_w, p_h) = PlayerCamera._xy_to_screen(view_size, (self.player.width, self.player.height), view_range, self.player.pos, absolute=True)
        player_pos = PlayerCamera._xy_to_screen(view_size, self.player.pos, view_range, self.player.pos)
        pygame.draw.polygon(screen, "red", [
            (player_pos[0] + forward[0] * p_h / 2 + left[0] * p_w / 2, player_pos[1] + forward[1] * p_h / 2 + left[1] * p_w / 2),
            (player_pos[0] + forward[0] * p_h / 2 - left[0] * p_w / 2, player_pos[1] + forward[1] * p_h / 2 - left[1] * p_w / 2),
            (player_pos[0] - forward[0] * p_h / 2 - left[0] * p_w / 2, player_pos[1] - forward[1] * p_h / 2 - left[1] * p_w / 2),
            (player_pos[0] - forward[0] * p_h / 2 + left[0] * p_w / 2, player_pos[1] - forward[1] * p_h / 2 + left[1] * p_w / 2)
        ])
        pygame.draw.line(screen, "black", player_pos, (player_pos[0] + forward[0] * p_h / 2, player_pos[1] + forward[1] * p_h / 2), 4)
        
    def input(self, dt):
        """
        Handle keyboard inputs
        """
        ZOOM_SPEED = 2.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_p]:
            self.ZOOM += ZOOM_SPEED * dt
        if keys[pygame.K_m]:
            self.ZOOM -= ZOOM_SPEED * dt
    
    def _xy_to_screen(view_size, pos, view_range, player_pos, absolute=False):
        """
        Convert a position in the grid to a position on the screen.
        
        Args:
            view_size: the size of the screen
            pos: the position in the grid (x, y) in [0, 1]
            view_range: the range of the view
            player_pos: the position of the player
            absolute: if True, the position is given in absolute coordinates
            
        Returns:
            the position on the screen
        """
        view_range = min(view_size) * view_range
        if not absolute:
            return (
                (pos[0] - player_pos[0] + 0.5) * view_range + (view_size[0] - view_range) / 2,
                (pos[1] - player_pos[1] + 0.5) * view_range + (view_size[1] - view_range) / 2
            )
        return (
            pos[0] * view_range,
            pos[1] * view_range
        )
        
    def _screen_to_xy(view_size, pos, view_range, player_pos):
        """
        Convert a position on the screen to a position in the grid.
        
        Args:
            view_size: the size of the screen
            pos: the position on the screen (x, y)
            view_range: the range of the view
            player_pos: the position of the player
            
        Returns:
            the position in the grid (x, y) in [0, 1]
        """
        view_range = min(view_size) * view_range
        return (
            (pos[0] - (view_size[0] - view_range) / 2) / view_range + player_pos[0] - 0.5,
            (pos[1] - (view_size[1] - view_range) / 2) / view_range + player_pos[1] - 0.5
        )