import pygame
import numpy as np
import random
import math
import json
import pickle
import base64
import os
import time

#Changing the current network from staeless to recurrent
#this will allow it to have a memory of past events

# Initialize Pygame
pygame.init()

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
APPLE_SIZE = 10
CREATURE_SIZE = 8
SCORE_ZONE_SIZE = 50
WORLD_PADDING = 20

# --- Simulation Parameters ---
# These can be overridden by the experiment-specific config
POPULATION_SIZE = 50
MAX_APPLES = 30
APPLE_RESPAWN_RATE = 0.5
STARTING_ENERGY = 100.0
ENERGY_DECAY_RATE = 0.1
GATHER_ENERGY_BONUS = 30.0
MAX_ENERGY = 200.0
GENERATION_TIME = 5000
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
ELITISM_COUNT = 2
HIDDEN_NODES = 10
USE_AUTO_GATHER = True 
USE_AUTO_FLEE = True      


def save_best_creature(creature, filename="best_creature.save"):
    """Save the best creature to a file"""
    save_data = {
        'weights': base64.b64encode(pickle.dumps(creature.nn.get_weights())).decode('utf-8'),
        'apples_deposited_total': creature.apples_deposited_total,
        'fitness': creature.fitness,
        'params': {
            'NUM_INPUTS': Creature.NUM_INPUTS,
            'NUM_OUTPUTS': Creature.NUM_OUTPUTS,
            'HIDDEN_NODES': HIDDEN_NODES
        }
    }
    with open(filename, 'w') as f:
        json.dump(save_data, f)
    print(f"Saved best creature to {filename}")


def load_best_creature(filename="best_creature.save"):
    """Load a saved creature from file"""
    try:
        with open(filename, 'r') as f:
            save_data = json.load(f)

        weights = pickle.loads(base64.b64decode(save_data['weights'].encode('utf-8')))
        # MODIFICATION: Use saved parameters for NN creation
        nn = NeuralNetwork(save_data['params']['NUM_INPUTS'], save_data['params']['HIDDEN_NODES'], save_data['params']['NUM_OUTPUTS'])
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

# --- Neural Network Class ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_h = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, output_size))

    def predict(self, inputs):
        inputs = np.array(inputs).reshape(1, -1)
        hidden_raw = np.dot(inputs, self.weights_ih) + self.bias_h
        hidden_activated = np.tanh(hidden_raw)
        output_raw = np.dot(hidden_activated, self.weights_ho) + self.bias_o
        output_activated = np.tanh(output_raw)
        return output_activated[0]

    def get_weights(self):
        return np.concatenate([self.weights_ih.flatten(), self.weights_ho.flatten()])

    def set_weights(self, flat_weights):
        ih_size = self.weights_ih.size
        ho_size = self.weights_ho.size
        if len(flat_weights) != ih_size + ho_size:
             raise ValueError("Incorrect number of weights provided")
        self.weights_ih = flat_weights[:ih_size].reshape(self.weights_ih.shape)
        self.weights_ho = flat_weights[ih_size:].reshape(self.weights_ho.shape)

    def mutate(self, rate, strength):
         weights = self.get_weights()
         for i in range(len(weights)):
             if random.random() < rate:
                 weights[i] += np.random.randn() * strength
         self.set_weights(weights)

# --- Simulation Object Classes ---
class Creature:
    # MODIFICATION: Increased NN inputs and outputs for new senses and actions.
    NUM_INPUTS = 11
    NUM_OUTPUTS = 5

    def __init__(self, x, y, nn):
        self.pos = np.array([x, y], dtype=np.float64)
        self.nn = nn
        self.energy = STARTING_ENERGY
        self.max_energy = MAX_ENERGY
        self.velocity = np.array([0.0, 0.0], dtype=np.float64)
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.1
        self.apples_held = 0
        self.max_apples = 5
        self.fitness = 0.0
        self.apples_deposited_total = 0
        self.lifetime = 0
        self.sight_radius = 150
        # MODIFICATION: Attributes for interaction and communication
        self.color = (0, 255, 0)
        self.communication_signal = 0.0

    def get_nearest_object(self, objects, ignore_self=False):
        """Finds the nearest object in a list to this creature."""
        if not objects:
            return None, self.sight_radius, 0.0
        if ignore_self:
            objects = [obj for obj in objects if obj is not self]
            if not objects: return None, self.sight_radius, 0.0
        positions = np.array([obj.pos for obj in objects])
        dist_sq = np.sum((positions - self.pos)**2, axis=1)
        min_idx = np.argmin(dist_sq)
        if dist_sq[min_idx] < self.sight_radius**2:
            dist = np.sqrt(dist_sq[min_idx])
            direction = positions[min_idx] - self.pos
            target_angle = math.atan2(direction[1], direction[0])
            relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            return objects[min_idx], dist, relative_angle / math.pi
        return None, self.sight_radius, 0.0

    def get_inputs(self, apples, score_zone, enemies, all_creatures):
        """Prepares the input vector for the neural network."""
        inputs = np.zeros(self.NUM_INPUTS)
        _, dist_apple, angle_apple = self.get_nearest_object(apples)
        inputs[0] = (self.sight_radius - dist_apple) / self.sight_radius
        inputs[1] = angle_apple
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
        zone_relative_angle = (zone_target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        inputs[2] = max(0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH)
        inputs[3] = zone_relative_angle / math.pi
        inputs[4] = self.energy / self.max_energy
        inputs[5] = self.apples_held / self.max_apples
        _, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        inputs[6] = (self.sight_radius - dist_enemy) / self.sight_radius
        inputs[7] = angle_enemy
        nearest_ally, dist_ally, angle_ally = self.get_nearest_object(all_creatures, ignore_self=True)
        inputs[8] = (self.sight_radius - dist_ally) / self.sight_radius
        inputs[9] = angle_ally
        inputs[10] = nearest_ally.communication_signal if nearest_ally else 0.0
        return inputs

    def update(self, apples, score_zone, enemies, all_creatures, dt=1.0):
        """Main update logic for the creature."""
        inputs = self.get_inputs(apples, score_zone, enemies, all_creatures)
        actions = self.nn.predict(inputs)
        turn_request, accel_request, _, deposit_request, comms_request = actions
        self.communication_signal = comms_request
        if self.communication_signal > 0.5: self.color = (255, 69, 0)
        elif self.communication_signal < -0.5: self.color = (0, 191, 255)
        else: self.color = (50, 205, 50)
        if USE_AUTO_GATHER:
            for apple in apples[:]:
                if np.linalg.norm(self.pos - apple.pos) < (CREATURE_SIZE + APPLE_SIZE):
                    apples.remove(apple)
                    self.apples_held = min(self.apples_held + 1, self.max_apples)
                    self.energy = min(self.energy + GATHER_ENERGY_BONUS, self.max_energy)
                    break
        if USE_AUTO_FLEE:
            nearest_enemy, dist_enemy, _ = self.get_nearest_object(enemies)
            if nearest_enemy and dist_enemy < 30:
                direction = self.pos - nearest_enemy.pos
                self.angle = math.atan2(direction[1], direction[0])
                accel_request = 1.0
        if deposit_request > 0.5 and self.apples_held > 0:
            if np.linalg.norm(self.pos - score_zone.pos) < (SCORE_ZONE_SIZE + CREATURE_SIZE):
                self.fitness += self.apples_held * 10
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0
        self.energy -= ENERGY_DECAY_RATE * dt
        if self.energy > 0:
            self.angle = (self.angle + turn_request * self.max_turn_rate) % (2 * math.pi)
            self.speed = min(self.speed + accel_request * 0.2, self.max_speed) if accel_request > 0 else self.speed * 0.95
            self.velocity = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed
            self.pos = np.clip(self.pos + self.velocity * dt, WORLD_PADDING, [SCREEN_WIDTH - WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING])
        self.lifetime += dt
        return self.energy > 0

    def draw(self, screen):
        energy_ratio = max(0, self.energy / MAX_ENERGY)
        energy_bar_width = int(CREATURE_SIZE * 2 * energy_ratio)
        energy_bar_pos = self.pos - np.array([CREATURE_SIZE, CREATURE_SIZE + 4])
        pygame.draw.rect(screen, (255, 0, 0), (*energy_bar_pos.astype(int), CREATURE_SIZE * 2, 3))
        pygame.draw.rect(screen, (0, 255, 0), (*energy_bar_pos.astype(int), energy_bar_width, 3))

        # MODIFICATION: Use self.color to draw the creature
        pygame.draw.circle(screen, self.color, self.pos.astype(int), CREATURE_SIZE)
        end_line = self.pos + np.array([math.cos(self.angle), math.sin(self.angle)]) * (CREATURE_SIZE + 5)
        pygame.draw.line(screen, (200, 200, 200), self.pos.astype(int), end_line.astype(int), 2)
        if self.apples_held > 0:
            font = pygame.font.SysFont(None, 18)
            text = font.render(str(self.apples_held), True, (255, 255, 255))
            screen.blit(text, (self.pos[0] - 4, self.pos[1] - 6))

class Enemy:
    def __init__(self, x, y, speed=2.0, sight_radius=120):
        self.pos = np.array([x, y], dtype=np.float64)
        self.speed = speed
        self.attack_damage = 20.0
        self.sight_radius = sight_radius

    def update(self, creatures):
        if not creatures:
            return
        # Only target creatures within sight radius
        visible_creatures = [c for c in creatures if np.linalg.norm(self.pos - c.pos) <= self.sight_radius]
        if not visible_creatures:
            return
        target = min(visible_creatures, key=lambda c: np.linalg.norm(self.pos - c.pos))
        direction = target.pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 1:
            self.pos += (direction / dist) * self.speed
        if dist < CREATURE_SIZE + 10:
            target.energy = max(0, target.energy - self.attack_damage)

    def draw(self, screen):
        pygame.draw.circle(screen, (200, 0, 50), self.pos.astype(int), CREATURE_SIZE + 2)
        # Optionally draw sight radius for visualization
        # pygame.draw.circle(screen, (200, 0, 50, 40), self.pos.astype(int), int(self.sight_radius), 1)

class Apple:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)
    def draw(self, screen): pygame.draw.circle(screen, (255, 0, 0), self.pos.astype(int), APPLE_SIZE)

class ScoreZone:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)
    def draw(self, screen): pygame.draw.circle(screen, (0, 0, 255, 100), self.pos.astype(int), SCORE_ZONE_SIZE)

def reproduction(parents, generation):
    next_population = []
    # Elitism
    parents.sort(key=lambda c: c.fitness, reverse=True)
    for i in range(min(ELITISM_COUNT, len(parents))):
        nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES, Creature.NUM_OUTPUTS)
        nn.set_weights(parents[i].nn.get_weights())
        next_population.append(Creature(random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50), nn))
    # Crossover and Mutation
    while len(next_population) < POPULATION_SIZE:
        p1, p2 = random.choices(parents, k=2)
        p1_w, p2_w = p1.nn.get_weights(), p2.nn.get_weights()
        cross_pt = random.randint(1, len(p1_w) - 1)
        child_w = np.concatenate([p1_w[:cross_pt], p2_w[cross_pt:]])
        nn = NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES, Creature.NUM_OUTPUTS)
        nn.set_weights(child_w)
        nn.mutate(MUTATION_RATE, MUTATION_STRENGTH)
        next_population.append(Creature(random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50), nn))
    return next_population

# --- Game Setup ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Emergent AI - Communicating Collectors')
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# MODIFICATION: Define four major food zones for apples to spawn in.
food_zone_size = 120
q_width, q_height = SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4
food_zones = [
    (q_width - food_zone_size / 2, q_width + food_zone_size / 2, q_height - food_zone_size / 2, q_height + food_zone_size / 2),
    (SCREEN_WIDTH - q_width - food_zone_size / 2, SCREEN_WIDTH - q_width + food_zone_size / 2, q_height - food_zone_size / 2, q_height + food_zone_size / 2),
    (q_width - food_zone_size / 2, q_width + food_zone_size / 2, SCREEN_HEIGHT - q_height - food_zone_size / 2, SCREEN_HEIGHT - q_height + food_zone_size / 2),
    (SCREEN_WIDTH - q_width - food_zone_size / 2, SCREEN_WIDTH - q_width + food_zone_size / 2, SCREEN_HEIGHT - q_height - food_zone_size / 2, SCREEN_HEIGHT - q_height + food_zone_size / 2)
]
def spawn_apple_in_zone():
    zone = random.choice(food_zones)
    x = random.uniform(zone[0], zone[1])
    y = random.uniform(zone[2], zone[3])
    return Apple(x, y)

score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
# MODIFICATION: Spawn initial apples within the defined zones.
apples = [spawn_apple_in_zone() for _ in range(MAX_APPLES)]
population = [Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                       random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                       NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES, Creature.NUM_OUTPUTS)) for _ in range(POPULATION_SIZE)]
enemies = [Enemy(300, 300), Enemy(500, 400)]

# --- Game Loop ---
running = True
frame_count = 0
generation = 1
all_time_best_fitness = -float('inf')

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and population:
                best_creature = max(population, key=lambda c: c.fitness)
                save_best_creature(best_creature)
            elif event.key == pygame.K_l:
                loaded = load_best_creature()
                if loaded: population.append(loaded) # Add loaded creature to population

    frame_count += 1
    if len(apples) < MAX_APPLES and random.random() < APPLE_RESPAWN_RATE:
        # MODIFICATION: Respawn apples within the defined zones.
        apples.append(spawn_apple_in_zone())

    # Update creatures and enemies
    for i in range(len(population) - 1, -1, -1):
        # MODIFICATION: Pass the entire population list for creature-to-creature sensing
        if not population[i].update(apples, score_zone, enemies, population):
            population.pop(i)
    for enemy in enemies: enemy.update(population)

    # Check for generation end
    if frame_count >= GENERATION_TIME or not population:
        print(f"--- Generation {generation} Complete ---")
        if population:
            best_gen_fitness = max(c.fitness for c in population)
            avg_fitness = sum(c.fitness for c in population) / len(population)
            print(f"  Creatures remaining: {len(population)}")
            print(f"  Best fitness: {best_gen_fitness:.2f}, Avg fitness: {avg_fitness:.2f}")
            if best_gen_fitness > all_time_best_fitness:
                all_time_best_fitness = best_gen_fitness
            population = reproduction(population, generation)
        else:
            print("  Population died out. Restarting...")
            population = [Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH-WORLD_PADDING),
                                   random.randint(WORLD_PADDING, SCREEN_HEIGHT-WORLD_PADDING),
                                   NeuralNetwork(Creature.NUM_INPUTS, HIDDEN_NODES, Creature.NUM_OUTPUTS)) for _ in range(POPULATION_SIZE)]
        frame_count = 0
        generation += 1

    # --- Drawing ---
    screen.fill((30, 30, 30))
    score_zone.draw(screen)
    for apple in apples: apple.draw(screen)
    for creature in population: creature.draw(screen)
    for enemy in enemies: enemy.draw(screen)

    # Draw Info Text
    gen_text = font.render(f"Generation: {generation}", True, (255, 255, 255))
    pop_text = font.render(f"Population: {len(population)}", True, (255, 255, 255))
    time_text = font.render(f"Time: {frame_count}/{GENERATION_TIME}", True, (255, 255, 255))
    best_text = font.render(f"All-Time Best Fitness: {all_time_best_fitness:.2f}", True, (255, 255, 0))
    screen.blit(gen_text, (10, 10))
    screen.blit(pop_text, (10, 30))
    screen.blit(time_text, (10, 50))
    screen.blit(best_text, (10, 70))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
