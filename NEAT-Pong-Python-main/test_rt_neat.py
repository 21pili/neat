import pygame
from pong import Game
import neat
import os
import pickle
import numpy as np
import graphviz
import warnings
import matplotlib.pyplot as plt

import math
import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean



class PongGame_rt:
    def __init__(self, window, width, height,g_mode):
        self.game=Game(window, width, height,g_mode)
        self.left_paddle=self.game.left_paddle
        self.right_paddle=self.game.right_paddle
        self.ball=self.game.ball
    
    def test_ai(self,genome,config,human=True):
        net=neat.nn.FeedForwardNetwork.create(genome,config)

        past_ball_x=self.game.ball.x
        past_ball_y=self.game.ball.y

        run=True
        clock=pygame.time.Clock()
        while run:
            clock.tick(80)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    run=False
                    break
            #print(self.game.ball.x,self.game.ball.y)
            if human:
                keys=pygame.key.get_pressed()
                if keys[pygame.K_UP]:
                    self.game.move_paddle(left=True, up=True)
                if keys[pygame.K_DOWN]:
                    self.game.move_paddle(left=True, up=False)

                output = net.activate((self.right_paddle.y, self.ball.y,
                                            abs(self.right_paddle.x-self.ball.x), self.left_paddle.y))
                
                decision=output.index(max(output))
                
                if decision==0:
                    pass
                elif decision==1:
                    self.game.move_paddle(left=False, up=True)
                else:
                    self.game.move_paddle(left=False, up=False)


                game_info=self.game.loop()
                self.game.draw(True,False)
                pygame.display.update()
            else:
                ball_x=self.game.ball.x
                ball_y=self.game.ball.y 

                output = net.activate((self.left_paddle.y, self.ball.y,
                                            abs(self.left_paddle.x-self.ball.x),self.right_paddle.y))
                
                decision=output.index(max(output))
                
                if decision==0:
                    pass
                elif decision==1:
                    self.game.move_paddle(left=True, up=True)
                else:
                    self.game.move_paddle(left=True, up=False)


                decision_opti=play_optimal(700,1000,past_ball_x,past_ball_y,ball_x,ball_y,self.right_paddle.y,self.right_paddle.HEIGHT)
                if decision_opti==0:
                    pass
                elif decision_opti==1:
                    self.game.move_paddle(left=False, up=True)
                else:
                    self.game.move_paddle(left=False, up=False)

                past_ball_x=ball_x
                past_ball_y=ball_y

                game_info=self.game.loop()
                self.game.draw(True,False)
                pygame.display.update()


        pygame.quit()

    def step_genome(self,config, genome):
        """
        Handles the evolution of the population
        """
        net=neat.nn.FeedForwardNetwork.create(genome,config)

        past_ball_x=self.game.ball.x
        past_ball_y=self.game.ball.y

        run=True
        while run:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()

            ball_x=self.game.ball.x
            ball_y=self.game.ball.y 

            output = net.activate((self.left_paddle.y, self.ball.y,
                                        abs(self.left_paddle.x-self.ball.x), self.right_paddle.y))
            
            decision=output.index(max(output))
            
            if decision==0:
                pass
            elif decision==1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)


            decision_opti=play_optimal(700,500,past_ball_x,past_ball_y,ball_x,ball_y,self.right_paddle.y,self.right_paddle.HEIGHT)
            if decision_opti==0:
                pass
            elif decision_opti==1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            past_ball_x=ball_x
            past_ball_y=ball_y

            game_info=self.game.loop()

            #self.game.draw(draw_score=False, draw_hits=True)
            #pygame.display.update()

            if game_info.right_hits>20:
                f=genome.fitness
                r=1.2
                genome.fitness+=(game_info.left_hits-game_info.right_score/2-f)/r
                if game_info.left_score>2:
                    genome.fitness+=0#game_info.left_score/2
                break
    
def train_ai(window,width,height,config,genomes,reproduce,g_mode):
    for genome in genomes:
        game=PongGame_rt(window,width,height,g_mode)
        game.step_genome(config,genome)
    return reproduce(config,genomes)


        
    
def play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,right_paddle_y,right_paddle_height):

    if ball_x>past_ball_x:
        target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
        if target_y_raw//height%2==1:
            target_y=height-target_y_raw%height
        else:
            target_y=target_y_raw%height
        #print(target_y-right_paddle_y)
        if target_y<right_paddle_y+right_paddle_height/2:
            #print('up')
            return(1)
        elif target_y>right_paddle_y+right_paddle_height/2:
            #print('down')
            return(2)
        else:
        #print('stay')
            return(0)
        
    else:
        if right_paddle_y+right_paddle_height/2>height/2:
            return(1)
        if right_paddle_y+right_paddle_height/2<height/2:
            return(2)
        else:
            return(0)
        


local_dir=os.path.dirname(__file__)
config_path=os.path.join(local_dir, "config.txt")

config=neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction
                            ,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)




def create_new(genome_type, genome_config, num_genomes):
    new_genomes = []
    genome_indexer=0
    for i in range(num_genomes):
        key = genome_indexer+1
        g = genome_type(key)
        g.configure_new(genome_config)
        g.fitness=0#-len(g.connections)/20
        #g.fitness-=len(g.nodes)/10
        new_genomes.append(g)

    return np.array(new_genomes)



def reproduce(config, old_population): 
    """
    Handles the reproduction of genomes,
    """
    fitnesses = np.array([g.fitness for g in old_population])

    

    old_population_sort=old_population[np.argsort(fitnesses)]
    
    """ parent1, parent2=old_population_sort[0],old_population_sort[0]
    while parent1==old_population_sort[0]:
        parent1 = random.choice(old_population)
    
    while parent2==old_population_sort[0] or parent2==parent1:
        parent2 = random.choice(old_population) """
    
    parent1, parent2=old_population_sort[-1],old_population_sort[-2]

    child = config.genome_type(0)
    child.configure_crossover(parent1, parent2, config.genome_config)
    child.mutate(config.genome_config)
    child.fitness=0#-len(child.connections)/20
    #child.fitness-=len(child.nodes)/10
    old_population_sort[0]=child

    return old_population_sort



def test_optimal(g_mode=False):
    width, height = 700, 1000
    window=pygame.display.set_mode((width, height))
    game =PongGame_rt(window, width, height,g_mode)
    
    run=True
    clock=pygame.time.Clock()

    past_ball_x=game.ball.x
    past_ball_y=game.ball.y

    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                run=False
                break
        
        ball_x=game.ball.x
        ball_y=game.ball.y
    
        if ball_x>past_ball_x:
            target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
            if target_y_raw//height%2==1:
                target_y=height-target_y_raw%height
            else:
                target_y=target_y_raw%height
    
            if target_y<game.right_paddle.y+game.right_paddle.HEIGHT/2:
                game.game.move_paddle(left=False, up=True)
            if target_y>game.right_paddle.y+game.right_paddle.HEIGHT/2:
                game.game.move_paddle(left=False, up=False)
        else:
            if game.right_paddle.y+game.right_paddle.HEIGHT/2>height/2:
                game.game.move_paddle(left=False, up=True)
            if game.right_paddle.y+game.right_paddle.HEIGHT/2<height/2:
                game.game.move_paddle(left=False, up=False)
        past_ball_x=ball_x
        past_ball_y=ball_y


        keys=pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            game.game.move_paddle(left=True, up=True)
        if keys[pygame.K_DOWN]:
            game.game.move_paddle(left=True, up=False)

        game.game.move_paddle(left=False, up=True)
        game.game.move_paddle(left=False, up=False)

        game.game.loop()
        game.game.draw(True,False)
        pygame.display.update()
    
    pygame.quit()


def test_ai(config,human=True,g_mode=False):

    width, height = 700, 1000
    window=pygame.display.set_mode((width, height))

    with open("best_rt.pickle","rb") as f:
        winner=pickle.load(f)
    
    game =PongGame_rt(window, width, height,g_mode)
    game.test_ai(winner,config,human)

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            val_weight='%.3f'%(cg.weight)
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width},label=str(val_weight))

    dot.render(filename, view=view)

    return dot

creation =True
test=True
display_net=True



if creation:
    genome_indexer=0
    new_genomes=create_new(config.genome_type, config.genome_config, config.pop_size)
    for genome in new_genomes:
        genome.fitness=0

    width,height=700,1000
    window=pygame.display.set_mode((width,height))



    for i in range(25):
        print(i)
        new_genomes=train_ai(window,width,height,config,new_genomes,reproduce,g_mode=False)
        print(max([g.fitness for g in new_genomes]))
        
    winner=new_genomes[np.argmax([g.fitness for g in new_genomes])]
    with open("best_rt.pickle","wb") as f:
            pickle.dump(winner, f)

if display_net:
    with open("best_rt.pickle","rb") as f:
        winner=pickle.load(f)
    draw_net(config,winner,view=True,filename='winner.gv')

if test:
    test_ai(config,human=False,g_mode=False)



#test_optimal(False)

