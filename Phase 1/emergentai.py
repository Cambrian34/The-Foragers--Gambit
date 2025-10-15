import pygame
import numpy as np
import random
import math
import json
import pickle
import base64
import os
import time
pygame.init()


#would placing the food in a specific area change the population drop rate 
#would adding perceptrons to see the distance to the enemies cause a longer time to converge. Right now its using 

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
APPLE_SIZE = 10  # Smaller apples
CREATURE_SIZE = 8 # Smaller creatures
SCORE_ZONE_SIZE = 50
WORLD_PADDING = 20

STATS_HISTORY_LENGTH = 100 


# Simulation Parameters
POPULATION_SIZE = 50
MAX_APPLES = 30
APPLE_RESPAWN_RATE = 0.5 # Chance per frame an apple respawns if below max
STARTING_ENERGY = 100.0
ENERGY_DECAY_RATE = 0.1 # Energy lost per frame just existing
MOVE_ENERGY_COST = 0.2  # Extra energy cost for moving
GATHER_ENERGY_BONUS = 30.0 # Energy gained per apple gathered
MAX_ENERGY = 200.0
REST_ENERGY_REGAIN = 0.5

# Genetic Algorithm Parameters
GENERATION_TIME = 5000 # Frames per generation 
MUTATION_RATE = 0.2  # Probability of a weight mutating
MUTATION_STRENGTH = 0.8 # How much a weight can change during mutation
ELITISM_COUNT = 2 # Keep the best N creatures without mutation

def save_best_creature(creature, filename="best_creature.save"):
    """Save the best creature to a file"""
    save_data = {
        'weights': base64.b64encode(pickle.dumps(creature.nn.get_weights())).decode('utf-8'),
        'apples_deposited_total': creature.apples_deposited_total,
        'fitness': creature.fitness,
        'params': {
            'NUM_INPUTS': Creature.NUM_INPUTS,
            'NUM_OUTPUTS': Creature.NUM_OUTPUTS
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f)
    print(f"Saved best creature to {filename}")

def save_simulation_state(generation, filename="simulation_state.save"):
    """Save the current simulation state"""
    if not hasattr('all_time_best', globals()):
        return
        
    state = {
        'generation': generation,
        'best_creature': base64.b64encode(pickle.dumps(all_time_best)).decode('utf-8'),
        'best_fitness': all_time_best_fitness
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(state, f)
    print(f"Saved simulation state to {filename}")

def load_best_creature(filename="best_creature.save"):
    """Load a saved creature from file"""
    try:
        with open(filename, 'r') as f:
            save_data = json.load(f)
        
        weights = pickle.loads(base64.b64decode(save_data['weights'].encode('utf-8')))
        nn = NeuralNetwork(save_data['params']['NUM_INPUTS'], 20, save_data['params']['NUM_OUTPUTS'])
        nn.set_weights(weights)
        
        creature = Creature(
            random.randint(50, SCREEN_WIDTH - 50),
            random.randint(50, SCREEN_HEIGHT - 50),
            nn
        )
        creature.apples_deposited_total = save_data.get('apples_deposited_total', 0)
        creature.fitness = save_data.get('fitness', 0)
        
        print(f"Loaded creature from {filename}")
        return creature
    except Exception as e:
        print(f"Failed to load creature: {e}")
        return None

def load_simulation_state(filename="simulation_state.save"):
    """Load the entire simulation state"""
    try:
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        global all_time_best, all_time_best_fitness
        all_time_best = pickle.loads(base64.b64decode(state['best_creature']))
        all_time_best_fitness = state['best_fitness']
        
        print(f"Loaded simulation state from {filename}")
        return state['generation']
    except Exception as e:
        print(f"Failed to load simulation state: {e}")
        return 1  # Return generation 1 if loading fails

# --- Neural Network --- using a feedforward neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.1
        # Simple biases (can be evolved too, but starting simple)
        self.bias_h = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, output_size))

    def predict(self, inputs):
        # Ensure inputs is a 2D array (1 sample, N features)
        inputs = np.array(inputs).reshape(1, -1)

        # Input to Hidden layer
        hidden_raw = np.dot(inputs, self.weights_ih) + self.bias_h
        hidden_activated = np.tanh(hidden_raw) # Use tanh activation function (-1 to 1)

        # Hidden to Output layer
        output_raw = np.dot(hidden_activated, self.weights_ho) + self.bias_o
        output_activated = np.tanh(output_raw) # Use tanh activation function (-1 to 1)

        return output_activated[0] # Return the 1D array of outputs

    def get_weights(self):
        # Return a flat list of all weights for mutation/crossover
        return np.concatenate([self.weights_ih.flatten(), self.weights_ho.flatten()])

    def set_weights(self, flat_weights):
        # Set weights from a flat list
        ih_size = self.weights_ih.size
        ho_size = self.weights_ho.size

        if len(flat_weights) != ih_size + ho_size:
             raise ValueError("Incorrect number of weights provided")

        self.weights_ih = flat_weights[:ih_size].reshape(self.weights_ih.shape)
        self.weights_ho = flat_weights[ih_size:].reshape(self.weights_ho.shape)

    def mutate(self, rate, strength):
         # Mutate weights slightly
         weights = self.get_weights()
         for i in range(len(weights)):
             if random.random() < rate:
                 weights[i] += np.random.randn() * strength
         self.set_weights(weights)

def draw_world_boundaries(screen):
    boundary_color = (100, 100, 100, 150)  # Semi-transparent gray
    boundary_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    
    # Draw four rectangles around the edges
    pygame.draw.rect(boundary_surface, boundary_color, 
                    (0, 0, SCREEN_WIDTH, WORLD_PADDING))
    pygame.draw.rect(boundary_surface, boundary_color,
                    (0, SCREEN_HEIGHT - WORLD_PADDING, SCREEN_WIDTH, WORLD_PADDING))
    pygame.draw.rect(boundary_surface, boundary_color,
                    (0, 0, WORLD_PADDING, SCREEN_HEIGHT))
    pygame.draw.rect(boundary_surface, boundary_color,
                    (SCREEN_WIDTH - WORLD_PADDING, 0, WORLD_PADDING, SCREEN_HEIGHT))
    
    screen.blit(boundary_surface, (0, 0))
# --- Creature Class ---
class Creature:
    
    NUM_INPUTS = 8  # or whatever the correct number of inputs is
    NUM_OUTPUTS = 4 
    
    # INPUTS indices (for the neural network)
    IDX_DIST_APPLE = 0
    IDX_ANGLE_APPLE = 1
    IDX_DIST_ZONE = 2
    IDX_ANGLE_ZONE = 3
    IDX_ENERGY = 4

    IDX_APPLES_HELD = 5
    IDX_DIST_ENEMY = 6  
    IDX_ANGLE_ENEMY = 7

    # OUTPUTS indices (for the neural network)
    IDX_TURN = 0
    IDX_ACCELERATE = 1
    IDX_GATHER = 2
    IDX_DEPOSIT = 3


    def __init__(self, x, y, nn):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.1

        self.energy = STARTING_ENERGY
        self.apples_held = 0
        self.max_apples = 5
        self.nn = nn
        self.fitness = 0.0
        self.apples_deposited_total = 0
        self.lifetime = 0  # Track how long the creature has lived

        self.sight_radius = 150

    def get_nearest_object(self, objects):
        nearest_obj = None
        min_dist_sq = self.sight_radius**2

        for obj in objects:
            direction = obj.pos - self.pos
            dist_sq = np.dot(direction, direction)

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_obj = obj

        if nearest_obj:
            dist = math.sqrt(min_dist_sq)
            direction_vec = nearest_obj.pos - self.pos
            target_angle = math.atan2(direction_vec[1], direction_vec[0])
            relative_angle = target_angle - self.angle
            relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi
            return nearest_obj, dist, relative_angle / math.pi
        else:
            return None, self.sight_radius, 0.0

    def get_inputs(self, apples, score_zone):
        inputs = np.zeros(Creature.NUM_INPUTS)

        # Nearest Apple Info
        nearest_apple, dist_apple, angle_apple = self.get_nearest_object(apples)
        inputs[Creature.IDX_DIST_APPLE] = (self.sight_radius - dist_apple) / self.sight_radius
        inputs[Creature.IDX_ANGLE_APPLE] = angle_apple

#*********new
        
        #get nearest enemy info
        nearest_enemy, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        inputs[Creature.IDX_DIST_ENEMY] = (self.sight_radius - dist_enemy) / self.sight_radius
        inputs[Creature.IDX_ANGLE_ENEMY] = angle_enemy
        #*******************
        

        # Score Zone Info
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
        zone_relative_angle = zone_target_angle - self.angle
        zone_relative_angle = (zone_relative_angle + math.pi) % (2 * math.pi) - math.pi

        inputs[Creature.IDX_DIST_ZONE] = max(0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH)
        inputs[Creature.IDX_ANGLE_ZONE] = zone_relative_angle / math.pi

        # Energy Level
        inputs[Creature.IDX_ENERGY] = self.energy / MAX_ENERGY

        #inputs[Creature.IDX_APPLES_HELD] = self.apples_held / self.max_apples  # Normalized


        return inputs

    def update(self, apples, score_zone,enemy_pos, dt=1.0):
        inputs = self.get_inputs(apples, score_zone)
        actions = self.nn.predict(inputs)

        turn_request = actions[Creature.IDX_TURN]
        accel_request = actions[Creature.IDX_ACCELERATE]
        gather_request = actions[Creature.IDX_GATHER]
        deposit_request = actions[Creature.IDX_DEPOSIT]
        
        if gather_request > 0.5:  # Adjust the gather condition to be more realistic
            nearby_apples = [a for a in apples 
                        if np.linalg.norm(self.pos - a.pos) < (CREATURE_SIZE + APPLE_SIZE)*2]
            for apple in nearby_apples:
                dist_sq = np.dot(self.pos - apple.pos, self.pos - apple.pos)
                if dist_sq < (CREATURE_SIZE + APPLE_SIZE)**2 and apple in apples:
                    apples.remove(apple)
                    self.apples_held += 1
                    self.energy += GATHER_ENERGY_BONUS
                    self.energy = min(self.energy, MAX_ENERGY)
                    if self.apples_held >= self.max_apples:
                        break
        

        
        nearest_enemy, dist_enemy, _ = self.get_nearest_object(enemies)

        if enemy_pos is not None:
            dist_to_enemy = np.linalg.norm(self.pos - enemy_pos)
            if dist_to_enemy < 25:
                self.energy -= 1.0  # Lose energy if too close to enemy
                self.energy = max(self.energy, 0)
                

        


        for apple in apples[:]:  # Iterate over a copy to allow removal
                distance = np.linalg.norm(self.pos - apple.pos)
                if distance < (CREATURE_SIZE + APPLE_SIZE) * 1.2:  # Slightly larger than sum of radii
                    apples.remove(apple)
                    self.apples_held = min(self.apples_held + 1, self.max_apples)
                    self.energy = min(self.energy + GATHER_ENERGY_BONUS, MAX_ENERGY)
                    break 
        

        if deposit_request > 0.5 and self.apples_held > 0:
            dist_sq_zone = np.dot(self.pos - score_zone.pos, self.pos - score_zone.pos)
            if dist_sq_zone < (SCORE_ZONE_SIZE + CREATURE_SIZE)**2:
                self.fitness += self.apples_held * 10
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0

        # Energy management
        #self.energy -= (ENERGY_DECAY_RATE) * dt

        # Movement and energy
        self.energy -= ENERGY_DECAY_RATE * dt
        if self.energy > 0:
            self.angle += turn_request * self.max_turn_rate
            self.angle %= (2 * math.pi)
    
            if accel_request > 0:
                self.speed = min(self.speed + accel_request * 0.2, self.max_speed)
            else:
                self.speed *= 0.95
    
            self.speed = max(0, self.speed)
    
            self.velocity[0] = math.cos(self.angle) * self.speed
            self.velocity[1] = math.sin(self.angle) * self.speed
            
            new_pos = self.pos + self.velocity * dt
            
            # Check and handle boundary collisions
            bounced = False
            if new_pos[0] < WORLD_PADDING:
                new_pos[0] = WORLD_PADDING
                self.angle = math.pi - self.angle
                bounced = True
            elif new_pos[0] > SCREEN_WIDTH - WORLD_PADDING:
                new_pos[0] = SCREEN_WIDTH - WORLD_PADDING
                self.angle = math.pi - self.angle
                bounced = True
            
            if new_pos[1] < WORLD_PADDING:
                new_pos[1] = WORLD_PADDING
                self.angle = -self.angle
                bounced = True
            elif new_pos[1] > SCREEN_HEIGHT - WORLD_PADDING:
                new_pos[1] = SCREEN_HEIGHT - WORLD_PADDING
                self.angle = -self.angle
                bounced = True
            
            if bounced:
                self.speed *= 0.8  # Lose some energy when bouncing
                self.velocity[0] = math.cos(self.angle) * self.speed
                self.velocity[1] = math.sin(self.angle) * self.speed
            
            self.pos = new_pos
    
        self.lifetime += dt
        return self.energy > 0
    

    def draw(self, screen):
        #pygame.draw.circle(screen, (0, 150, 0), self.pos.astype(int), CREATURE_SIZE)

        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 5)
        pygame.draw.line(screen, (0, 255, 0), self.pos.astype(int), end_line.astype(int), 2)

        color = (0, min(255, 100 + self.apples_deposited_total * 20), 0)

        energy_ratio = max(0, self.energy / MAX_ENERGY)
        energy_bar_width = int(CREATURE_SIZE * 2 * energy_ratio)
        energy_bar_pos = self.pos - np.array([CREATURE_SIZE, CREATURE_SIZE + 4])
        pygame.draw.rect(screen, (255, 0, 0), (*energy_bar_pos.astype(int), CREATURE_SIZE * 2, 3))
        pygame.draw.rect(screen, (0, 255, 0), (*energy_bar_pos.astype(int), energy_bar_width, 3))
        pygame.draw.circle(screen, color, self.pos.astype(int), CREATURE_SIZE)

        save_text = font.render("Press S to save, L to load", True, (200, 200, 200))
        screen.blit(save_text, (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 30))

        if saving:
            save_status = font.render("Saved successfully!", True, (0, 255, 0))
            screen.blit(save_status, (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 60))
        elif loading:
            load_status = font.render("Loaded successfully!", True, (0, 255, 0))
            screen.blit(load_status, (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 60))


        if self.apples_held > 0:
            if not hasattr(self, 'font'):
                self.font = pygame.font.SysFont(None, 18)  # Initialize font once
            text = self.font.render(str(self.apples_held), True, (255, 255, 255))
            screen.blit(text, (self.pos[0] - 4, self.pos[1] - 6))

# --- Enemy Class ---
class Enemy:
    def __init__(self, x, y, speed=2.0):
        self.pos = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = speed
        self.max_speed = 1.5
        self.health = 10.0
        self.attack_damage = .1
        self.sight_radius = 50

    def move_towards(self, target_pos):
        direction = target_pos - self.pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance  # Normalize the vector
        self.velocity = direction * self.speed
        self.pos += self.velocity

    def update(self, creatures):
        nearest_creature, _, _ = self.get_nearest_object(creatures)
        if nearest_creature:
            self.move_towards(nearest_creature.pos)
            # Simple attack behavior if in range
            if np.linalg.norm(self.pos - nearest_creature.pos) < CREATURE_SIZE + 10:
                nearest_creature.energy -= self.attack_damage  # Attack the creature
                nearest_creature.energy = max(nearest_creature.energy, 0)  # Prevent negative energy

    def get_nearest_object(self, creatures):
        nearest_obj = None
        min_dist_sq = self.sight_radius**2
        for obj in creatures:
            direction = obj.pos - self.pos
            dist_sq = np.dot(direction, direction)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_obj = obj
        return nearest_obj, np.sqrt(min_dist_sq), 0.0

    def draw(self, screen):
        pygame.draw.circle(screen, (128, 0, 128), self.pos.astype(int), CREATURE_SIZE)  # Purple color


# --- Apple Class ---
class Apple:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), self.pos.astype(int), APPLE_SIZE)

# --- ScoreZone Class ---
class ScoreZone:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float64)

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 255, 100), self.pos.astype(int), SCORE_ZONE_SIZE) # Make semi-transparent


# --- Genetic Algorithm Functions ---
def selection(population):
    # Sort population by fitness (descending)
    population.sort(key=lambda creature: creature.fitness, reverse=True)
    return population[:POPULATION_SIZE // 4] # Select the top 25% as parents

def reproduction(parents, generation):
    next_population = []

    # Elitism: Keep the best N creatures unchanged
    for i in range(ELITISM_COUNT):
         if i < len(parents):
            child_nn = NeuralNetwork(Creature.NUM_INPUTS, 10, Creature.NUM_OUTPUTS) # 10 hidden nodes
            child_nn.set_weights(parents[i].nn.get_weights()) # Copy weights directly
            child = Creature(random.randint(50, SCREEN_WIDTH - 50),
                           random.randint(50, SCREEN_HEIGHT - 50),
                           child_nn)
            child.fitness = 0 # Reset fitness for new generation
            next_population.append(child)


    # Crossover and Mutation
    while len(next_population) < POPULATION_SIZE:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents) # Could be the same parent

        # Create child NN and inherit weights (simple average crossover)
        # More complex crossover could be used
        p1_weights = parent1.nn.get_weights()
        p2_weights = parent2.nn.get_weights()
        # Simple averaging (could also do single-point crossover)
        # child_weights = (p1_weights + p2_weights) / 2
        # Single Point Crossover:
        crossover_point = random.randint(1, len(p1_weights) - 1)
        child_weights = np.concatenate([p1_weights[:crossover_point], p2_weights[crossover_point:]])


        child_nn = NeuralNetwork(Creature.NUM_INPUTS, 10, Creature.NUM_OUTPUTS) # Ensure correct architecture
        child_nn.set_weights(child_weights)

        # Mutate the child's NN
        child_nn.mutate(MUTATION_RATE, MUTATION_STRENGTH)

        # Create the new creature
        child = Creature(random.randint(50, SCREEN_WIDTH - 50),
                         random.randint(50, SCREEN_HEIGHT - 50),
                         child_nn)
        next_population.append(child)

    return next_population

# --- Game Setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Emergent AI - Apple Collectors')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Initialize objects
score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
apples = [Apple(random.randint(APPLE_SIZE, SCREEN_WIDTH - APPLE_SIZE),
                random.randint(APPLE_SIZE, SCREEN_HEIGHT - APPLE_SIZE)) for _ in range(MAX_APPLES)]

#population initialization:
population = []
for _ in range(POPULATION_SIZE):
    nn = NeuralNetwork(Creature.NUM_INPUTS, 10, Creature.NUM_OUTPUTS)
    creature = Creature(
        random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
        random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
        nn
    )
    population.append(creature)

# Initialize enemies
enemy1 = Enemy(300, 300)
enemy2 = Enemy(500, 400)
enemies = [enemy1, enemy2]

# Update in the main loop:
def update_world():
    for enemy in enemies:
        enemy.update(population)  # Let enemies interact with creatures

    for creature in population:
        nearest_enemy, _, _ = creature.get_nearest_object(enemies)
        enemy_pos = nearest_enemy.pos if nearest_enemy else None
        creature.update(apples, score_zone, enemy_pos)

# Draw enemies in the world:
def draw_world():
    for enemy in enemies:
        enemy.draw(screen)
# --- Game Loop ---
running = True
frame_count = 0
generation = 1

all_time_best = None
all_time_best_fitness = -float('inf')
saving = False
loading = False
last_save_time = 0

"""
# Load previous state if available
if os.path.exists("simulation_state.save"):
    generation = load_simulation_state() + 1
    print(f"Resuming simulation from generation {generation}")
"""

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Auto-save on quit
            if all_time_best:
                save_best_creature(all_time_best)
                save_simulation_state(generation)
            running = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                if all_time_best:
                    save_best_creature(all_time_best)
                    save_simulation_state(generation)
                    saving = True
                    last_save_time = pygame.time.get_ticks()
            elif event.key == pygame.K_l:
                loaded = load_best_creature()
                if loaded:
                    population.sort(key=lambda c: c.fitness)
                    population[0] = loaded
                    loading = True
                    last_save_time = pygame.time.get_ticks()
    
    # Clear save/load messages after 2 seconds
    if pygame.time.get_ticks() - last_save_time > 2000:  # Clear messages after 2 seconds
        saving = False
        loading = False

    # --- Update Logic ---
    frame_count += 1

    # Respawn apples
    if len(apples) < MAX_APPLES and random.random() < APPLE_RESPAWN_RATE * (MAX_APPLES - len(apples)):
         apples.append(Apple(random.randint(APPLE_SIZE, SCREEN_WIDTH - APPLE_SIZE),
                           random.randint(APPLE_SIZE, SCREEN_HEIGHT - APPLE_SIZE)))

    # Update creatures
    next_population_indices = list(range(len(population))) # Indices of creatures alive
    for i in range(len(population) - 1, -1, -1): # Iterate backwards for safe removal
        creature = population[i]
        # def update(self, apples, score_zone,enemy_pos, dt=1.0):
        enemy_pos = enemies[i % len(enemies)].pos if enemies else None
        alive = creature.update(apples, score_zone, enemy_pos, dt=1.0)
        if not alive:
            # Optional: Could add a penalty to fitness upon death
            # creature.fitness -= 50
            population.pop(i)
            next_population_indices.pop(i) # remove index if creature died

    # Check if generation time is up or population died out
    if len(population) == 0 or frame_count >= GENERATION_TIME:
        print(f"--- Generation {generation} Complete ---")
        if len(population) > 0:
             # Calculate average and best fitness
             total_fitness = sum(c.fitness for c in population)
             avg_fitness = total_fitness / len(population)
             best_fitness = max(c.fitness for c in population)
             total_deposited = sum(c.apples_deposited_total for c in population)
             print(f"  Creatures remaining: {len(population)}")
             print(f"  Best fitness: {best_fitness:.2f}")
             print(f"  Avg fitness: {avg_fitness:.2f}")
             print(f"  Total apples deposited this gen: {total_deposited}")


             # Genetic Algorithm Steps
             parents = selection(population)
             population = reproduction(parents, generation)
        else:
            print("  Population died out. Restarting with random creatures.")
             # Re-initialize population if all died
            population = []
            for _ in range(POPULATION_SIZE):
                nn = NeuralNetwork(Creature.NUM_INPUTS, 10, Creature.NUM_OUTPUTS)
                creature = Creature(random.randint(50, SCREEN_WIDTH - 50),
                                    random.randint(50, SCREEN_HEIGHT - 50),
                                    nn)
                population.append(creature)

        # Reset for next generation
        frame_count = 0
        generation += 1
        # Keep existing apples for the new generation
        # Optional: Reset apples?
        # apples = [Apple(random.randint(APPLE_SIZE, SCREEN_WIDTH - APPLE_SIZE),
        #                 random.randint(APPLE_SIZE, SCREEN_HEIGHT - APPLE_SIZE)) for _ in range(MAX_APPLES)]


    # --- Drawing ---
    screen.fill((30, 30, 30))  # Dark background

    # Draw score zone first (background)
    score_zone.draw(screen)

    # Draw apples
    for apple in apples:
        apple.draw(screen)

    # Draw creatures
    for creature in population:
        creature.draw(screen)

    if len(population) > 0:
        stats = {
            'generation': generation,
            'population': len(population),
            'best_fitness': max(c.fitness for c in population),
            'avg_fitness': sum(c.fitness for c in population) / len(population),
            'total_apples': sum(c.apples_deposited_total for c in population),
            'avg_lifetime': sum(c.lifetime for c in population) / len(population)
        }

    # Draw Info Text
    gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
    pop_text = font.render(f"Population: {len(population)}", True, (255, 255, 255))
    time_text = font.render(f"Time: {frame_count}/{GENERATION_TIME}", True, (255, 255, 255))
    apple_text = font.render(f"Apples: {len(apples)}/{MAX_APPLES}", True, (255, 255, 255)) 
    highest_fitness = max((c.fitness for c in population), default=0) 
    if highest_fitness > all_time_best_fitness:
        all_time_best_fitness = highest_fitness
        all_time_best = max(population, key=lambda c: c.fitness)
    if all_time_best:
        best_text = font.render(f"All-Time Best Fitness: {all_time_best_fitness:.2f}", True, (255, 255, 0))
        screen.blit(best_text, (10, 90))

    screen.blit(gen_text, (10, 10))
    screen.blit(pop_text, (10, 30))
    screen.blit(time_text, (10, 50))
    screen.blit(apple_text, (10, 70))  

    # Draw enemies
    update_world()

    # Draw the world (including creatures and enemies)
    draw_world()


    # Update display
    pygame.display.flip()
    clock.tick(60)  # Target 60 FPS

pygame.quit()