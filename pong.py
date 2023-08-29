import pygame
import sys
import random
import neat
import pickle
import math

TRAIN_MODE = True  # Set to True to train, False to use the best genome
MAX_TIME = 45

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 900, 750

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Constants
BALL_DIMENSION = 15
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
BALL_SPEED = 4
PADDLE_SPEED = 4
FPS = 3000

class SaveBestGenome(neat.reporting.BaseReporter):
    def __init__(self):
        super().__init__()
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        max_fitness = float("-inf")
        best_genome = None

        for species_id, species in species_set.species.items():
            for genome_id, genome in species.members.items():
                if genome.fitness is not None and genome.fitness > max_fitness:
                    max_fitness = genome.fitness
                    best_genome = genome

        if best_genome:
            with open(f"best_genome.pkl", "wb") as f:
                pickle.dump(best_genome, f)


def save_best_genome(genomes):
    max_fitness = float("-inf")
    best_genome = None

    for genome_id, genome in genomes:
        if genome.fitness is not None and genome.fitness > max_fitness:
            max_fitness = genome.fitness
            best_genome = genome

    if best_genome:
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(best_genome, f)


class PongEnv:
    def __init__(self, genome_a, genome_b, config, generation=0, genome_id=0, max_fitness=0, fitness=0):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pong")
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()

        # Create networks for both genomes
        self.net_a = neat.nn.FeedForwardNetwork.create(genome_a, config)
        if genome_b:
            self.net_b = neat.nn.FeedForwardNetwork.create(genome_b, config)
        else:
            self.net_b = None

        self.generation = generation
        self.genome_id = genome_id
        self.fitness = fitness if fitness is not None else 0.0
        self.max_fitness = max_fitness if max_fitness is not None else 0.0

        self.ball = pygame.Rect(WIDTH // 2 - BALL_DIMENSION // 2, HEIGHT // 2 - BALL_DIMENSION // 2, BALL_DIMENSION, BALL_DIMENSION)
        self.paddle_a = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.paddle_b = pygame.Rect(WIDTH - 30, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)

        self.ball_speed = [BALL_SPEED, BALL_SPEED]
        self.font = pygame.font.SysFont(None, 35)

        self.score_a = 0
        self.score_b = 0

    def get_elapsed_time(self):
        current_time = pygame.time.get_ticks()
        elapsed_time = (current_time - self.start_time) / 1000.0  # Convert milliseconds to seconds
        return elapsed_time
    
    def reset_ball(self):
        self.ball.x = WIDTH // 2 - BALL_DIMENSION // 2
        self.ball.y = HEIGHT // 2 - BALL_DIMENSION // 2
        self.ball_speed = [BALL_SPEED * random.choice([-1, 1]), BALL_SPEED * random.choice([-1, 1])]

    def update(self):
        # Move ball
        self.ball.move_ip(self.ball_speed)

        # Ball collision with top and bottom
        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_speed[1] = -self.ball_speed[1]

        # Ball collision with paddles
        if self.ball.colliderect(self.paddle_a) and self.ball_speed[0] < 0:
            if self.ball.top > self.paddle_a.top and self.ball.bottom < self.paddle_a.bottom:
                self.ball_speed[0] = -self.ball_speed[0]
        elif self.ball.colliderect(self.paddle_b) and self.ball_speed[0] > 0:
            if self.ball.top > self.paddle_b.top and self.ball.bottom < self.paddle_b.bottom:
                self.ball_speed[0] = -self.ball_speed[0]

        # Ball out of bounds
        if self.ball.left <= 0 or self.ball.right >= WIDTH:
            if self.ball.left <= 0:
                self.score_b += 1
            else:
                self.score_a += 1
            self.reset_ball()

        # AI movement for paddle A
        ball_x, ball_y = self.ball.center
        ball_x_vel, ball_y_vel = self.ball_speed
        paddle_a_y = self.paddle_a.centery
        output_a = self.net_a.activate((ball_x, ball_y, ball_x_vel, ball_y_vel, paddle_a_y))
        if output_a[0] > 0.5 and self.paddle_a.top > 0:
            self.paddle_a.move_ip(0, -PADDLE_SPEED)
        elif output_a[1] > 0.5 and self.paddle_a.bottom < HEIGHT:
            self.paddle_a.move_ip(0, PADDLE_SPEED)

        # AI movement for paddle B
        if self.net_b:
            paddle_b_y = self.paddle_b.centery
            output_b = self.net_b.activate((ball_x, ball_y, ball_x_vel, ball_y_vel, paddle_b_y))
            if output_b[0] > 0.5 and self.paddle_b.top > 0:
                self.paddle_b.move_ip(0, -PADDLE_SPEED)
            elif output_b[1] > 0.5 and self.paddle_b.bottom < HEIGHT:
                self.paddle_b.move_ip(0, PADDLE_SPEED)

    def render(self):
        self.screen.fill(BLACK)
        pygame.draw.ellipse(self.screen, WHITE, self.ball)
        pygame.draw.rect(self.screen, WHITE, self.paddle_a)
        pygame.draw.rect(self.screen, WHITE, self.paddle_b)

        # Draw score
        score_text = self.font.render(f"AI: {self.score_a} - Player: {self.score_b}", True, WHITE)
        self.screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 10))

        # Render game details
        details_text = self.font.render(f"Gen: {self.generation} - Genome: {self.genome_id} - Max Fitness: {self.max_fitness:.2f}", True, WHITE)
        self.screen.blit(details_text, (10, score_text.get_height() + 20))  # Position below score text

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        pygame.display.quit()
        sys.exit()

def eval_genomes(genomes, config):
    max_fitness = genomes[0][1].fitness if genomes else 0  # Assume sorted order

    # Convert genomes to list to easily access pairs
    genome_list = list(genomes)
    

    for i in range(0, len(genome_list), 2):  # Pair genomes in twos
        genome_a_id, genome_a = genome_list[i]
        genome_b_id, genome_b = genome_list[i+1] if i+1 < len(genome_list) else (None, None)  # Ensure there's a genome for paddle B

        potential_generation = math.ceil(genome_a_id/50)-1
        
        env = PongEnv(genome_a, genome_b, config, generation=potential_generation, genome_id=genome_a_id, fitness=genome_a.fitness, max_fitness=max_fitness)

        genome_a.fitness = 0  # Start with fitness level of 0
        if genome_b:
            genome_b.fitness = 0

        start_time = pygame.time.get_ticks()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    if TRAIN_MODE:
                        save_best_genome(genomes)
                    env.close()

            env.update()  # All movement logic is handled within the environment now
            env.render()

            # Increase the fitness for survival
            genome_a.fitness += 0.1
            if genome_b:
                genome_b.fitness += 0.1

            elapsed_time = (pygame.time.get_ticks() - start_time) / 1000.0  # Convert to seconds

            # End the game for the genome if either player reaches 15 points
            if elapsed_time == MAX_TIME or env.score_a >= 15 or env.score_b >= 15:
                break

            # Adjusting fitness values depending on the game's outcome can be more nuanced.
            if env.score_a > env.score_b:
                genome_a.fitness += 1
            elif env.score_b > env.score_a:
                if genome_b is not None:
                    genome_b.fitness += 1



def play_best_genome(net):
    env = PongEnv(best_genome,best_genome,config)
    start_time = pygame.time.get_ticks()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()

        # Feed neural network the inputs and get the output
        ball_x, ball_y = env.ball.center
        ball_x_vel, ball_y_vel = env.ball_speed
        player_paddle_y = env.paddle_b.centery

        output = net.activate((ball_x, ball_y, ball_x_vel, ball_y_vel, player_paddle_y))

        # Depending on the output, move the paddle up or down
        if output[0] > 0.5 and env.paddle_b.top > 0:
            env.paddle_b.move_ip(0, -PADDLE_SPEED)
        elif output[1] > 0.5 and env.paddle_b.bottom < HEIGHT:
            env.paddle_b.move_ip(0, PADDLE_SPEED)

        env.update()
        env.render()

        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000.0

        # End the game if either player reaches 15 points
        if env.score_a >= 15 or env.score_b >= 15:
            break

if __name__ == "__main__":
    # Load the NEAT config file
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if TRAIN_MODE:
        # If in training mode, create the population and run the NEAT algorithm
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(SaveBestGenome())  # Add this line

        winner = p.run(eval_genomes, 50)  # Run for 50 generations

        # Save the best genome
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(winner, f)
    else:
        # If not in training mode, load the best genome and use it
        with open("best_genome.pkl", "rb") as f:
            best_genome = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        play_best_genome(net)
