import pygame
from pong import Game
import neat
import os
import pickle
import numpy as np

from visualize import Visualize
draw_net=Visualize.draw_net




class PongGame_rt:
    def __init__(self, window, width, height):
        self.game=Game(window, width, height)
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
                                            abs(self.right_paddle.x-self.ball.x)))#, self.left_paddle.y))
                
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
                                            abs(self.left_paddle.x-self.ball.x)))#,self.right_paddle.y))
                
                decision=output.index(max(output))
                
                if decision==0:
                    pass
                elif decision==1:
                    self.game.move_paddle(left=True, up=True)
                else:
                    self.game.move_paddle(left=True, up=False)


                decision_opti=play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,self.right_paddle.y,self.right_paddle.HEIGHT)
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
    
def create_new(genome_type, genome_config, num_genomes):
    new_genomes = []
    genome_indexer=0
    for i in range(num_genomes):
        key = genome_indexer+1
        g = genome_type(key)
        g.configure_new(genome_config)
        g.fitness=0
        new_genomes.append(g)

    return np.array(new_genomes)

def step_genome(window,width,height,config, genome):
    """
    Handles the evolution of the population
    """
    game=PongGame_rt(window,width,height)
    net=neat.nn.FeedForwardNetwork.create(genome,config)

    past_ball_x=game.ball.x
    past_ball_y=game.ball.y
    past_left_hits=0
    past_right_hits=0
    s=0
    frac=np.random.uniform(0,1)


    run=True
    #clock=pygame.time.Clock()
    while run:
        #clock.tick(80)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()

        ball_x=game.ball.x
        ball_y=game.ball.y 

        output = net.activate((game.left_paddle.y, game.ball.y,
                                    abs(game.left_paddle.x-game.ball.x)))#, game.right_paddle.y))
        
        decision=output.index(max(output))
        
        if decision==0:
            pass
        elif decision==1:
            game.game.move_paddle(left=True, up=True)
        else:
            game.game.move_paddle(left=True, up=False)


        decision_opti=play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,game.right_paddle.y,game.right_paddle.HEIGHT,frac)
        if decision_opti==0:
            pass
        elif decision_opti==1:
            game.game.move_paddle(left=False, up=True)
        else:
            game.game.move_paddle(left=False, up=False)

        
        
        
        game_info=game.game.loop()
        right_hits=game_info.right_hits
        left_hits=game_info.left_hits
        if right_hits>past_right_hits:
            frac=np.random.uniform(0,1)
        
        if left_hits>past_left_hits:
            target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
            if target_y_raw//height%2==1:
                target_y=height-target_y_raw%height
            else:
                target_y=target_y_raw%height

            right_paddle_y=game.right_paddle.y
            s+=abs(target_y-right_paddle_y-game.right_paddle.HEIGHT/2) / height   
        
        #game.game.draw(draw_score=False, draw_hits=True)
        #pygame.display.update()

        past_ball_x=ball_x
        past_ball_y=ball_y
        past_left_hits=left_hits
        past_right_hits=right_hits

        c_s,c_h,c_d=0,1,0
        r=1
        if game_info.right_hits>20:
            f=genome.fitness
            genome.fitness+=(c_s*s+c_h*game_info.left_hits+c_d*(game_info.left_score-game_info.right_score)-f)/r
            break
    
def reproduce(config, old_population): 
    """
    Handles the reproduction of genomes,
    """
    fitnesses = np.array([g.fitness for g in old_population])

    

    old_population_sort=old_population[np.argsort(fitnesses)]
    
    parent1, parent2=old_population_sort[-1],old_population_sort[-2]

    child = config.genome_type(0)
    child.configure_crossover(parent1, parent2, config.genome_config)
    child.mutate(config.genome_config)
    child.fitness=0
    old_population_sort[0]=child

    return old_population_sort
            
def train_ai(window,width,height,config,genomes):
    for genome in genomes:
        step_genome(window,width,height,config,genome)
    return reproduce(config,genomes)



def run_evolution(width,height,nb_generations):
    new_genomes=create_new(config.genome_type, config.genome_config, config.pop_size)
    for genome in new_genomes:
        genome.fitness=0

    window=pygame.display.set_mode((width,height))



    for i in range(nb_generations):
        print(i)
        new_genomes=train_ai(window,width,height,config,new_genomes)
        print(max([g.fitness for g in new_genomes]))
        
    winner=new_genomes[np.argmax([g.fitness for g in new_genomes])]
    with open("best_rt.pickle","wb") as f:
            pickle.dump(winner, f)



        
    
def play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,right_paddle_y,right_paddle_height,frac):
    if ball_x>past_ball_x:
        target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
        if target_y_raw//height%2==1:
            target_y=height-target_y_raw%height
        else:
            target_y=target_y_raw%height
        #print(target_y-right_paddle_y)
        if target_y<right_paddle_y+right_paddle_height*frac:
            #print('up')
            return(1)
        elif target_y>right_paddle_y+right_paddle_height*frac:
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



test=True
display_net=False
human=True
    
width, height = 700, 500
window=pygame.display.set_mode((width, height))


run_evolution(width,height,50)


if display_net:
    with open("best_rt.pickle","rb") as f:
        winner=pickle.load(f)
    draw_net(config,winner,view=True,filename='winner.gv')

if test:
    window=pygame.display.set_mode((width, height))

    with open("best_rt.pickle","rb") as f:
        winner=pickle.load(f)
    
    game =PongGame_rt(window, width, height)
    game.test_ai(winner,config,human)



#test_optimal(False)

