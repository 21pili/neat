import pygame

class Display:
    def __init__(self):
        self.font = pygame.font.Font(None, 23)
        self.draw_count_left = 0
        self.draw_count_right = 0
        
    def clear(self):
        self.draw_count_left = 0
        self.draw_count_right = 0

    def display(self, screen, dir, message):
        # Get text representation of the message
        text = self.font.render(message, True, (200, 60, 60))
        
        # Compute x position
        if dir == "LEFT":
            x = 10
            screen.blit(text, (x, 10 + self.draw_count_left * 30))
            self.draw_count_left += 1
        else:
            x = screen.get_width() - self.font.size(message)[0] - 10
            screen.blit(text, (x, 10 + self.draw_count_right * 30))
            self.draw_count_right += 1
        
        