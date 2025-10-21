
import numpy as np
import random
import math
import os
import csv
import json
import pickle
import base64

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WORLD_PADDING = 20
APPLE_SIZE = 10
CREATURE_SIZE = 8
SCORE_ZONE_SIZE = 50

# --- Default Simulation Parameters ---
# These can be overridden by the experiment-specific config
DEFAULT_CONFIG = {
    "population_size": 50,
    "max_apples": 30,
    "apple_respawn_rate": 0.5,
    "starting_energy": 100.0,
    "energy_decay_rate": 0.1,
    "gather_energy_bonus": 30.0,
    "max_energy": 200.0,
    "generation_time": 5000,
    "mutation_rate": 0.1,
    "mutation_strength": 0.5,
    "elitism_count": 2,
    "hidden_nodes": 10,
    "use_auto_gather": True, # Tweakable parameter
    "use_auto_flee": True      # Tweakable parameter
}

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

class Creature:
    # MODIFICATION: Increased NN inputs and outputs for new senses and actions.
    # Inputs: dist/angle to apple, dist/angle to score zone, energy, apples held,
    #         dist/angle to nearest enemy, dist/angle to nearest creature,
    #         and nearest creature's communication signal.
    # Outputs: turn, accelerate, gather, deposit, communicate.
    NUM_INPUTS = 11
    NUM_OUTPUTS = 5

    def __init__(self, x, y, nn, config):
        self.pos = np.array([x, y], dtype=np.float64)
        self.nn = nn
        self.config = config 
        self.energy = self.config['starting_energy']
        self.max_energy = self.config['max_energy']
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
        self.color = (0, 255, 0)  # Default color (green)
        self.communication_signal = 0.0  # Represents the creature's current signal

    def get_nearest_object(self, objects, ignore_self=False):
        """Finds the nearest object in a list to this creature."""
        if not objects:
            return None, self.sight_radius, 0.0

        # Filter out self if required
        if ignore_self:
            objects = [obj for obj in objects if obj is not self]
            if not objects:
                return None, self.sight_radius, 0.0

        positions = np.array([obj.pos for obj in objects])
        dist_sq = np.sum((positions - self.pos)**2, axis=1)
        min_idx = np.argmin(dist_sq)

        # Check if the nearest object is within sight radius
        if dist_sq[min_idx] < self.sight_radius**2:
            dist = np.sqrt(dist_sq[min_idx])
            direction = positions[min_idx] - self.pos
            target_angle = math.atan2(direction[1], direction[0])
            # Calculate relative angle normalized between -1 and 1
            relative_angle = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            return objects[min_idx], dist, relative_angle / math.pi

        return None, self.sight_radius, 0.0

    def get_inputs(self, apples, score_zone, enemies, all_creatures):
        """Prepares the input vector for the neural network."""
        inputs = np.zeros(self.NUM_INPUTS)

        # Inputs 0, 1: Nearest apple
        _, dist_apple, angle_apple = self.get_nearest_object(apples)
        inputs[0] = (self.sight_radius - dist_apple) / self.sight_radius
        inputs[1] = angle_apple

        # Inputs 2, 3: Score zone
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
        zone_relative_angle = (zone_target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        inputs[2] = max(0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH)
        inputs[3] = zone_relative_angle / math.pi

        # Inputs 4, 5: Internal state
        inputs[4] = self.energy / self.max_energy
        inputs[5] = self.apples_held / self.max_apples

        # MODIFICATION: New sensory inputs for creatures and enemies
        # Inputs 6, 7: Nearest enemy
        _, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        inputs[6] = (self.sight_radius - dist_enemy) / self.sight_radius
        inputs[7] = angle_enemy

        # Inputs 8, 9, 10: Nearest friendly creature and its signal
        nearest_ally, dist_ally, angle_ally = self.get_nearest_object(all_creatures, ignore_self=True)
        inputs[8] = (self.sight_radius - dist_ally) / self.sight_radius
        inputs[9] = angle_ally
        inputs[10] = nearest_ally.communication_signal if nearest_ally else 0.0

        return inputs

    def update(self, apples, score_zone, enemies, all_creatures, dt=1.0):
        """Main update logic for the creature."""
        inputs = self.get_inputs(apples, score_zone, enemies, all_creatures)
        actions = self.nn.predict(inputs)
        # MODIFICATION: Unpack new communication output from the neural network
        turn_request, accel_request, _, deposit_request, comms_request = actions

        # MODIFICATION: Update communication signal and color based on NN output
        self.communication_signal = comms_request
        if self.communication_signal > 0.5:
            self.color = (255, 69, 0)  # Red/Orange for danger/alert
        elif self.communication_signal < -0.5:
            self.color = (0, 191, 255) # Deep Sky Blue for food/safety
        else:
            self.color = (50, 205, 50) # Lime Green for neutral

        # Automatic behavior toggles (can be learned by NN to ignore/enhance)
        if self.config['use_auto_gather']:
            for apple in apples[:]:
                if np.linalg.norm(self.pos - apple.pos) < (CREATURE_SIZE + APPLE_SIZE):
                    apples.remove(apple)
                    self.apples_held = min(self.apples_held + 1, self.max_apples)
                    self.energy = min(self.energy + self.config['gather_energy_bonus'], self.max_energy)
                    break

        if self.config['use_auto_flee']:
            nearest_enemy, dist_enemy, _ = self.get_nearest_object(enemies)
            if nearest_enemy and dist_enemy < 30:
                direction = self.pos - nearest_enemy.pos # Flee direction
                self.angle = math.atan2(direction[1], direction[0])
                accel_request = 1.0 # Override to force flee

        # Action: Deposit apples
        if deposit_request > 0.5 and self.apples_held > 0:
            if np.linalg.norm(self.pos - score_zone.pos) < (SCORE_ZONE_SIZE + CREATURE_SIZE):
                self.fitness += self.apples_held * 10
                self.apples_deposited_total += self.apples_held
                self.apples_held = 0

        # Update energy and physics
        self.energy -= self.config['energy_decay_rate'] * dt
        if self.energy > 0:
            self.angle = (self.angle + turn_request * self.max_turn_rate) % (2 * math.pi)
            self.speed = min(self.speed + accel_request * 0.2, self.max_speed) if accel_request > 0 else self.speed * 0.95
            self.velocity = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed
            self.pos = np.clip(self.pos + self.velocity * dt, WORLD_PADDING, [SCREEN_WIDTH - WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING])

        self.lifetime += dt
        return self.energy > 0

class Enemy:
    def __init__(self, x, y, speed=2.0, sight_radius=60):
        self.pos = np.array([x, y], dtype=np.float64)
        self.speed = speed
        self.attack_damage = 20.0
        self.sight_radius = sight_radius

    def update(self, creatures):
        if not creatures:
            return
        # Find creatures within sight radius
        visible_creatures = [c for c in creatures if np.linalg.norm(self.pos - c.pos) <= self.sight_radius]
        if not visible_creatures:
            return  # No target in sight
        target = min(visible_creatures, key=lambda c: np.linalg.norm(self.pos - c.pos))
        direction = target.pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 1:
            self.pos += (direction / dist) * self.speed
        if dist < CREATURE_SIZE + 10:
            target.energy = max(0, target.energy - self.attack_damage)

class Apple:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)

class ScoreZone:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)


# --- Data Logging Functions ---
def setup_csv_logger(filename):
    header = ['generation', 'creatures_remaining', 'best_fitness', 'avg_fitness', 'total_apples_deposited']
    # Ensure the directory for the results file exists
    if not os.path.exists(os.path.dirname(filename)) and os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename))
    with open(filename, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

def log_to_csv(filename, stats_dict):
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=stats_dict.keys())
        writer.writerow(stats_dict)

# --- Headless Simulation Runner ---
def run_headless_simulation(config, total_generations, results_csv_path):
    cfg = {**DEFAULT_CONFIG, **config}
    setup_csv_logger(results_csv_path)

    # MODIFICATION: Defined four major food zones for apples to spawn in.
    # This created a more structured environment instead of random distribution.
    food_zone_size = 120
    q_width, q_height = SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4
    food_zones = [
        # Top-left zone
        (q_width - food_zone_size / 2, q_width + food_zone_size / 2,
         q_height - food_zone_size / 2, q_height + food_zone_size / 2),
        # Top-right zone
        (SCREEN_WIDTH - q_width - food_zone_size / 2, SCREEN_WIDTH - q_width + food_zone_size / 2,
         q_height - food_zone_size / 2, q_height + food_zone_size / 2),
        # Bottom-left zone
        (q_width - food_zone_size / 2, q_width + food_zone_size / 2,
         SCREEN_HEIGHT - q_height - food_zone_size / 2, SCREEN_HEIGHT - q_height + food_zone_size / 2),
        # Bottom-right zone
        (SCREEN_WIDTH - q_width - food_zone_size / 2, SCREEN_WIDTH - q_width + food_zone_size / 2,
         SCREEN_HEIGHT - q_height - food_zone_size / 2, SCREEN_HEIGHT - q_height + food_zone_size / 2)
    ]

    def spawn_apple_in_zone():
        """Helper function to spawn an apple in one of the four random zones."""
        zone = random.choice(food_zones)
        x = random.uniform(zone[0], zone[1])
        y = random.uniform(zone[2], zone[3])
        return Apple(x, y)


    score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    # MODIFICATION: Spawned initial apples within the defined zones.
    apples = [spawn_apple_in_zone() for _ in range(cfg['max_apples'])]
    population = [Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                           random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                           NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_nodes'], Creature.NUM_OUTPUTS),
                           cfg) for _ in range(cfg['population_size'])]
    enemies = [Enemy(300, 300), Enemy(500, 400)]

    for gen in range(1, total_generations + 1):
        frame_count = 0
        while frame_count < cfg['generation_time'] and population:
            frame_count += 1
            if len(apples) < cfg['max_apples'] and random.random() < cfg['apple_respawn_rate']:
                # MODIFICATION: Respawned apples within the defined zones.
                apples.append(spawn_apple_in_zone())
            # MODIFICATION: Looped backwards to safely remove dead creatures
            for i in range(len(population) - 1, -1, -1):
                creature = population[i]
                # MODIFICATION: Passed the entire population list for creature-to-creature sensing
                if not creature.update(apples, score_zone, enemies, population):
                    population.pop(i)
            for enemy in enemies: enemy.update(population)

        print(f"  Gen {gen}/{total_generations} complete for '{config.get('name', 'N/A')}'. Pop left: {len(population)}")
        if population:
            stats = {
                'generation': gen,
                'creatures_remaining': len(population),
                'best_fitness': f"{max(c.fitness for c in population):.2f}",
                'avg_fitness': f"{sum(c.fitness for c in population) / len(population):.2f}",
                'total_apples_deposited': sum(c.apples_deposited_total for c in population)
            }
            log_to_csv(results_csv_path, stats)

            population.sort(key=lambda c: c.fitness, reverse=True)
            parents = population[:max(1, cfg['population_size'] // 4)]
            next_pop = []
            for i in range(min(cfg['elitism_count'], len(parents))):
                nn = NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_nodes'], Creature.NUM_OUTPUTS)
                nn.set_weights(parents[i].nn.get_weights())
                next_pop.append(Creature(random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50), nn, cfg))
            while len(next_pop) < cfg['population_size']:
                p1, p2 = random.choices(parents, k=2)
                p1_w, p2_w = p1.nn.get_weights(), p2.nn.get_weights()
                cross_pt = random.randint(1, len(p1_w) - 1)
                child_w = np.concatenate([p1_w[:cross_pt], p2_w[cross_pt:]])
                nn = NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_nodes'], Creature.NUM_OUTPUTS)
                nn.set_weights(child_w)
                nn.mutate(cfg['mutation_rate'], cfg['mutation_strength'])
                next_pop.append(Creature(random.randint(50, SCREEN_WIDTH-50), random.randint(50, SCREEN_HEIGHT-50), nn, cfg))
            population = next_pop
        else:
            log_to_csv(results_csv_path, {'generation': gen, 'creatures_remaining': 0, 'best_fitness': 0, 'avg_fitness': 0, 'total_apples_deposited': 0})
            population = [Creature(random.randint(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING),
                                   random.randint(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING),
                                   NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_nodes'], Creature.NUM_OUTPUTS),
                                   cfg) for _ in range(cfg['population_size'])]

