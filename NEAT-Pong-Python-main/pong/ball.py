import pygame
import math
import random


class Ball:
    MAX_VEL = 5
    RADIUS = 7

    def __init__(self, x, y):
    
        self.g=0.05#random.gauss(0,1)*1e-2
        self.x_3, self.y_3, self.g3=0,0,0#random.gauss(0,1)*1e-4
        self.x = self.original_x = x
        self.y = self.original_y = y
        self.x_0=self.x
        self.y_0=self.y
        
        angle = self._get_random_angle(-5, 5, [0])
        
        pos = 1 if random.random() < 0.5 else -1

        self.x_vel = pos * abs(math.cos(angle) * self.MAX_VEL)
        self.y_vel = math.sin(angle) * self.MAX_VEL

    def _get_random_angle(self, min_angle, max_angle, excluded):
        angle = 0
        while angle in excluded:
            angle = math.radians(random.randrange(min_angle, max_angle))

        return angle

    def draw(self, win):
        pygame.draw.circle(win, (255, 255, 255), (self.x, self.y), self.RADIUS)

    def move(self,g_mode):
        
        if g_mode:
            self.y_vel+=self.g
    
        self.y+=self.y_vel
        
        self.x += self.x_vel
        
        
    def reset(self):
        self.x_3,self.y_3,self.g3=0,0,0#random.gauss(0,1)*1e-4


        self.x = self.original_x
        self.y = self.original_y

        #sign_x_vel = np.sign(self.x_vel)
        angle = self._get_random_angle(-5, 5, [0])
        #x_vel = abs(math.cos(angle) * self.MAX_VEL)
        y_vel = math.sin(angle) * self.MAX_VEL

        self.y_vel = y_vel
        #self.x_vel = -sign_x_vel * x_vel
        self.x_vel *=-1

        self.x_0=self.x
        self.y_0=self.y