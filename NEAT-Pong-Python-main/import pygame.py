import pygame
from pong import Game
import neat
import os
import pickle
import numpy as np
import graphviz
import warnings
import matplotlib.pyplot as plt


#def potential(x,y):
    #V=-100*np.exp((-(x-350)**2)/50**2-((y-250)**2)/50**2)
    #dV_x=-2*(x-350)/50**2*V
    #dV_y=-2*(y-250)/50**2*V
    #return(dV_x,dV_y)


class PongGame:
    def __init__(self, window, width, height, g_mode):
        self.game=Game(window, width, height, g_mode)
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
                                            abs(self.right_paddle.x-self.ball.x)))
                
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
                                            abs(self.left_paddle.x-self.ball.x)))
                
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
                self.game.draw(True,False)
                pygame.display.update()


        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        net1=neat.nn.FeedForwardNetwork.create(genome1,config)
        net2=neat.nn.FeedForwardNetwork.create(genome2,config)
        
        run=True
        while run:
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    pygame.quit()
                    
            output1 = net1.activate((self.left_paddle.y, self.ball.y,
                                      abs(self.left_paddle.x-self.ball.x)))
            decision1=output1.index(max(output1))
            
            if decision1==0:
                pass
            elif decision1==1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)
            
            output2 = net2.activate((self.right_paddle.y, self.ball.y,
                                        abs(self.right_paddle.x-self.ball.x)))
            
            decision2=output2.index(max(output2))
            
            if decision2==0:
                pass
            elif decision2==1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)


            

            game_info=self.game.loop()

            #self.game.draw(draw_score=False, draw_hits=True)
            #pygame.display.update()

            if game_info.left_score>=1 or game_info.right_score>=1 or game_info.left_hits>50:
                self.calculate_fitness(genome1, genome2, game_info)
                break
        
    def calculate_fitness(self,genome1,genome2, game_info):
        genome1.fitness+=game_info.left_hits
        genome2.fitness+=game_info.right_hits
        if game_info.left_hits>5:
            if game_info.left_score>game_info.right_score:
                genome1.fitness+=10
            else:
                genome2.fitness+=10
        

def eval_genomes(genomes, config, g_mode=False):
    width, height = 700, 500

    window=pygame.display.set_mode((width, height))

    for i, (genom_id1, genome1) in enumerate(genomes):
        if i==len(genomes)-1:
            break
        genome1.fitness=0
        #for j, (genomeid2, genome2) in enumerate(genomes):
        #    if i!=j:
        print(i)
        for genomeid2, genome2 in genomes[i+1:]:
                genome2.fitness=0 if genome2.fitness is None else genome2.fitness
                game=PongGame(window, width, height,g_mode)
                game.train_ai(genome1,genome2,config)
                
def run_neat(config,g_mode=False):
    #p=neat.Checkpointer.restore_checkpoint("neat-checkpoint-1")
    p=neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats=neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 50)

    with open("best.pickle","wb") as f:
        pickle.dump(winner, f)
    return(stats)

def test_ai(config,human=True,g_mode=False):
    width, height = 700, 500
    window=pygame.display.set_mode((width, height))

    with open("best.pickle","rb") as f:
        winner=pickle.load(f)
    
    game =PongGame(window, width, height,g_mode)
    game.test_ai(winner,config,human)


def test_optimal(g_mode=False):
    width, height = 700, 500
    window=pygame.display.set_mode((width, height))
    game =PongGame(window, width, height,g_mode)
    
    run=True
    clock=pygame.time.Clock()

    past_ball_x=game.ball.x
    past_ball_y=game.ball.y

    while run:
        clock.tick(80)
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
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width},label=val_weight)

        dot.render(filename, view=view)

        return dot

def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        avg_fitness = np.array(statistics.get_fitness_mean())
        stdev_fitness = np.array(statistics.get_fitness_stdev())

        plt.plot(generation, avg_fitness, 'b-', label="average")
        plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
        plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
        plt.plot(generation, best_fitness, 'r-', label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale('symlog')

        plt.savefig(filename)
        if view:
            plt.show()

        plt.close()

def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()




if __name__=="__main__":
    local_dir=os.path.dirname(__file__)
    config_path=os.path.join(local_dir, "config.txt")

    config=neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction
                              ,neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    #run_neat(config,g_mode=False)
    #plot_stats(stats, True, True, "avg_fitness.svg")
    #plot_species(stats, True, "speciation.svg")
    
    with open("best.pickle","rb") as f:
        winner=pickle.load(f)
    draw_net(config, winner, True, "winner.gv")
    test_ai(config,human=False,g_mode=True)
    #test_optimal(True)

#draw_net(config, pickle.load(open("best.pickle","rb")), True)



