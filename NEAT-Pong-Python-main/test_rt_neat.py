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
        """
        Initialization of the game, of the left and right paddles and of the ball
        """
        self.game=Game(window, width, height)
        self.left_paddle=self.game.left_paddle
        self.right_paddle=self.game.right_paddle
        self.ball=self.game.ball
    
    def test_ai(self,genome,config,human=True):
        """
        Allow to compare any genome to either a human or an optimal bot
        """
        net=neat.nn.FeedForwardNetwork.create(genome,config) #neural network which is chosen

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
            
            if human: #First we consider the case where the genome has to play against a human
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
            else: #Then we consider a game between a genome and a bot
                ball_x=self.game.ball.x
                ball_y=self.game.ball.y 

                output = net.activate((self.left_paddle.y, self.ball.y,
                                            abs(self.left_paddle.x-self.ball.x)))#,self.right_paddle.y))
                
                decision=output.index(max(output))
                
                if decision==0: #the paddle controled by the ai does not move
                    pass
                elif decision==1: #the paddle controled by the ai goes up
                    self.game.move_paddle(left=True, up=True)
                else: #the paddle controled by the ai goes down
                    self.game.move_paddle(left=True, up=False)


                decision_opti=play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,self.right_paddle.y,self.right_paddle.HEIGHT)
                if decision_opti==0: #the paddle controled by the bot does not move
                    pass
                elif decision_opti==1: #the paddle controled by the bot goes up
                    self.game.move_paddle(left=False, up=True)
                else: #the paddle controled by the bot goes down
                    self.game.move_paddle(left=False, up=False)

                past_ball_x=ball_x
                past_ball_y=ball_y

                game_info=self.game.loop()
                self.game.draw(True,False)
                pygame.display.update()


        pygame.quit()
    
def create_new(genome_type, genome_config, num_genomes):
    """
    This function creates a new random population, according to the configuration parameter of Neat
    """
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
    Handles the evaluation of a neural network : a pong game between an ai and an optimal bot
    """
    game=PongGame_rt(window,width,height)
    net=neat.nn.FeedForwardNetwork.create(genome,config)

    past_ball_x=game.ball.x
    past_ball_y=game.ball.y
    past_left_hits=0
    past_right_hits=0
    s=0 #each time the ai hits the ball, the distance between the opponent and the spot where the ball will hit the right side is added to s
    frac=np.random.uniform(0,1) #this random variable corresponds to the spots where the ball will hit the paddle controled by the bot, it will change each time the bot hits a ball


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
        
        if decision==0: #the paddle controled by the ai does not move
            pass
        elif decision==1: #the paddle controled by the ai goes up
            game.game.move_paddle(left=True, up=True)
        else: #the paddle controled by the ai goes down
            game.game.move_paddle(left=True, up=False)


        decision_opti=play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,game.right_paddle.y,game.right_paddle.HEIGHT,frac)
        if decision_opti==0: #the paddle controled by the bot does not move
            pass
        elif decision_opti==1: #the paddle controled by the bot goes up
            game.game.move_paddle(left=False, up=True)
        else: #the paddle controled by the bot goes down
            game.game.move_paddle(left=False, up=False)

        
        
        
        game_info=game.game.loop()
        right_hits=game_info.right_hits
        left_hits=game_info.left_hits
        if right_hits>past_right_hits:
            frac=np.random.uniform(0,1) #this random variable corresponds to the spots where the ball will hit the paddle controled by the bot
        
        if left_hits>past_left_hits:
            target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
            if target_y_raw//height%2==1:
                target_y=height-target_y_raw%height
            else:
                target_y=target_y_raw%height

            right_paddle_y=game.right_paddle.y
            s+=abs(target_y-right_paddle_y-game.right_paddle.HEIGHT/2) / height   

        #Uncomment the two following lines if you want to look at the training
        #game.game.draw(draw_score=False, draw_hits=True)
        #pygame.display.update()

        past_ball_x=ball_x
        past_ball_y=ball_y
        past_left_hits=left_hits
        past_right_hits=right_hits

        c_s,c_h,c_d=0,1,0 #some coefficients to play with to easily tweak the fitness : c_s corresponds to the variable s, c_h to the number of left hits and c_d to the difference in scores between the ai and the bot
        r=1 #this number handles the memory of the ai in rt_Neat, it set to 1, then the population of ai has no memory
        if game_info.right_hits>20:
            f=genome.fitness
            genome.fitness+=(c_s*s+c_h*game_info.left_hits+c_d*(game_info.left_score-game_info.right_score)-f)/r
            break
    
def reproduce(config, old_population): 
    """
    Handles the reproduction of genomes
    """
    fitnesses = np.array([g.fitness for g in old_population])

    

    old_population_sort=old_population[np.argsort(fitnesses)]
    
    parent1, parent2=old_population_sort[-1],old_population_sort[-2] #We choose as parents the best individuals of the previous generation

    child = config.genome_type(0) #initialization of the child
    child.configure_crossover(parent1, parent2, config.genome_config) #mating 
    child.mutate(config.genome_config) #mutation
    child.fitness=0 #initialization of the child's fitness
    old_population_sort[0]=child #the child replace the worst genome of the previous generation

    return old_population_sort
            
def train_ai(window,width,height,config,genomes):
    """
    Handles one full step of the training of the neural networks : evaluation + reproduction
    """
    for genome in genomes:
        step_genome(window,width,height,config,genome)
    return reproduce(config,genomes)



def run_evolution(width,height,nb_generations):
    """
    Handles all the steps of the evolution
    """
    
    #Initialization of the population
    new_genomes=create_new(config.genome_type, config.genome_config, config.pop_size)
    for genome in new_genomes:
        genome.fitness=0

    #Initialization of the game
    window=pygame.display.set_mode((width,height))


    #Evolution
    for i in range(nb_generations):
        print(i)
        new_genomes=train_ai(window,width,height,config,new_genomes)
        print(max([g.fitness for g in new_genomes]))

    #We keep the best genome
    winner=new_genomes[np.argmax([g.fitness for g in new_genomes])]
    with open("best_rt.pickle","wb") as f:
            pickle.dump(winner, f)



        
    
def play_optimal(width,height,past_ball_x,past_ball_y,ball_x,ball_y,right_paddle_y,right_paddle_height,frac):
    """
    Handles the reaction of an optimal player
    width: width of the window game
    height: height of the window game
    past_ball_x, past_ball_y: previous position of the ball
    ball_x, ball_y: current position of the ball
    right_paddle_y: current position of the right paddle, controlled by the bot
    right_paddle_height: height of the right paddle, controlled by the bot
    frac: fraction of the right paddle (i.e. a number between 0 and 1) where the paddle must hit the ball
    """
    #When the ball goes toward the right, the spot where the ball will hit the right side is calculated, hereinafter called the target
    if ball_x>past_ball_x:
        target_y_raw=((ball_y-past_ball_y)/(ball_x-past_ball_x)*(width-ball_x)+ball_y)
        if target_y_raw//height%2==1:
            target_y=height-target_y_raw%height
        else:
            target_y=target_y_raw%height

        #If the target is above the paddle spot where it should be hit, then the paddle goes up
        if target_y<right_paddle_y+right_paddle_height*frac:
            return(1)
        #Otherwise it goes down
        elif target_y>right_paddle_y+right_paddle_height*frac:
            return(2)
        #When the paddle has reached the target, it no longer moves
        else:
            return(0)
            
    #When the ball is going to the left, the right paddle gets to the middle of the window
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



test=True #set to True if you want to test the best ai, which has been obtained after the training
display_net=False #set to True if you want to display the network of the best ai
human=True #set to True if you want to play against the best ai during the test
    
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

