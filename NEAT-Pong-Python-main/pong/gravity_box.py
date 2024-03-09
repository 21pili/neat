
import pygame


class Gravity_box():
    def __init__(self, x, y, width, height):
        self.x=x-width//2
        self.y=y-height//2
        self.width=width
        self.height=height
    
    def draw(self, win):
        #draw only edges
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y, self.width, 5))
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y+self.height-5, self.width, 5))
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y, 5, self.height))
        pygame.draw.rect(win, (255, 255, 255), (self.x+self.width-5, self.y, 5, self.height))

        