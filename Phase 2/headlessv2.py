"""
Headless Simulation Core - Upgraded for Project Chimera Phase II

This version integrates the LSTM Neural Network and Tournament Selection
to facilitate the evolution of more complex, memory-based behaviors.
"""

import numpy as np
import random
import math
import os
import csv

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WORLD_PADDING = 20
APPLE_SIZE = 10
CREATURE_SIZE = 8
SCORE_ZONE_SIZE = 50

# --- Default Simulation Parameters ---
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
    "hidden_size": 12, # Size of the LSTM memory cell
    "tournament_size": 3, # Number of individuals in a selection tournament
    "use_auto_gather": True,
    "use_auto_flee": True
}

# --- LSTM Neural Network Class ---
# Integrated from 'emergentai copy 2.py'
class LSTM_NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size
        concat_size = input_size + hidden_size
        self.wf = np.random.randn(concat_size, hidden_size) * 0.1
        self.bf = np.zeros((1, hidden_size))
        self.wi = np.random.randn(concat_size, hidden_size) * 0.1
        self.bi = np.zeros((1, hidden_size))
        self.wc = np.random.randn(concat_size, hidden_size) * 0.1
        self.bc = np.zeros((1, hidden_size))
        self.wo = np.random.randn(concat_size, hidden_size) * 0.1
        self.bo = np.zeros((1, hidden_size))
        self.w_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros((1, output_size))
        self.hidden_state = np.zeros((1, hidden_size))
        self.cell_state = np.zeros((1, hidden_size))

    def reset_state(self):
        self.hidden_state = np.zeros((1, self.hidden_size))
        self.cell_state = np.zeros((1, self.hidden_size))

    def predict(self, inputs):
        inputs = np.array(inputs).reshape(1, -1)
        combined_input = np.concatenate((inputs, self.hidden_state), axis=1)
        f = 1 / (1 + np.exp(-(np.dot(combined_input, self.wf) + self.bf)))
        i = 1 / (1 + np.exp(-(np.dot(combined_input, self.wi) + self.bi)))
        c_candidate = np.tanh(np.dot(combined_input, self.wc) + self.bc)
        self.cell_state = f * self.cell_state + i * c_candidate
        o = 1 / (1 + np.exp(-(np.dot(combined_input, self.wo) + self.bo)))
        self.hidden_state = o * np.tanh(self.cell_state)
        output_raw = np.dot(self.hidden_state, self.w_out) + self.b_out
        return np.tanh(output_raw)[0]

    def get_weights(self):
        return np.concatenate([
            m.flatten() for m in [self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, self.w_out, self.b_out]
        ])

    def set_weights(self, flat_weights):
        s = 0
        def extract(shape):
            nonlocal s
            size = np.prod(shape)
            w = flat_weights[s : s + size].reshape(shape)
            s += size
            return w
        self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, self.w_out, self.b_out = [extract(m.shape) for m in [self.wf, self.bf, self.wi, self.bi, self.wc, self.bc, self.wo, self.bo, self.w_out, self.b_out]]

    def mutate(self, rate, strength):
        weights = self.get_weights()
        for i in range(len(weights)):
            if random.random() < rate:
                weights[i] += np.random.randn() * strength
        self.set_weights(weights)

# --- Simulation Object Classes ---
class Creature:
    NUM_INPUTS = 11
    NUM_OUTPUTS = 5

    def __init__(self, x, y, nn, config):
        self.pos = np.array([x, y], dtype=np.float64)
        self.nn = nn
        self.config = config
        self.energy = self.config['starting_energy']
        self.max_energy = self.config['max_energy']
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.max_speed = 3.0
        self.max_turn_rate = 0.1
        self.apples_held = 0
        self.max_apples = 5
        self.fitness = 0.0
        self.apples_deposited_total = 0
        self.sight_radius = 150
        self.communication_signal = 0.0

    def get_nearest_object(self, objects, ignore_self=False):
        if ignore_self: objects = [o for o in objects if o is not self]
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
        inputs = np.zeros(self.NUM_INPUTS)
        _, dist_apple, angle_apple = self.get_nearest_object(apples)
        inputs[0:2] = [(self.sight_radius - dist_apple) / self.sight_radius, angle_apple]
        zone_dir = score_zone.pos - self.pos
        dist_zone = np.linalg.norm(zone_dir)
        zone_target_angle = math.atan2(zone_dir[1], zone_dir[0])
        zone_relative_angle = (zone_target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        inputs[2:4] = [max(0, (SCREEN_WIDTH - dist_zone) / SCREEN_WIDTH), zone_relative_angle / math.pi]
        inputs[4:6] = [self.energy / self.max_energy, self.apples_held / self.max_apples]
        _, dist_enemy, angle_enemy = self.get_nearest_object(enemies)
        inputs[6:8] = [(self.sight_radius - dist_enemy) / self.sight_radius, angle_enemy]
        nearest_ally, dist_ally, angle_ally = self.get_nearest_object(all_creatures, ignore_self=True)
        inputs[8:11] = [(self.sight_radius - dist_ally) / self.sight_radius, angle_ally, nearest_ally.communication_signal if nearest_ally else 0.0]
        return inputs

    def update(self, apples, score_zone, enemies, all_creatures):
        inputs = self.get_inputs(apples, score_zone, enemies, all_creatures)
        actions = self.nn.predict(inputs)
        turn_request, accel_request, _, deposit_request, comms_request = actions
        self.communication_signal = comms_request
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
                direction = self.pos - nearest_enemy.pos
                self.angle = math.atan2(direction[1], direction[0])
                accel_request = 1.0
        if deposit_request > 0.5 and self.apples_held > 0 and np.linalg.norm(self.pos - score_zone.pos) < (SCORE_ZONE_SIZE + CREATURE_SIZE):
            self.fitness += self.apples_held * 10
            self.apples_deposited_total += self.apples_held
            self.apples_held = 0
        self.energy -= self.config['energy_decay_rate']
        if self.energy > 0:
            self.angle = (self.angle + turn_request * self.max_turn_rate) % (2 * math.pi)
            self.speed = min(self.speed + accel_request * 0.2, self.max_speed) if accel_request > 0 else self.speed * 0.95
            velocity = np.array([math.cos(self.angle), math.sin(self.angle)]) * self.speed
            self.pos = np.clip(self.pos + velocity, WORLD_PADDING, [SCREEN_WIDTH - WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING])
        return self.energy > 0

class Enemy:
    def __init__(self, x, y, speed=2.0):
        self.pos = np.array([x, y], dtype=np.float64)
        self.speed = speed
        self.attack_damage = 20.0
        self.sight_radius = 120

    def update(self, creatures):
        if not creatures: return
        visible = [c for c in creatures if np.linalg.norm(self.pos - c.pos) <= self.sight_radius]
        if not visible: return
        target = min(visible, key=lambda c: np.linalg.norm(self.pos - c.pos))
        direction = target.pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 1: self.pos += (direction / dist) * self.speed
        if dist < CREATURE_SIZE + 10: target.energy = max(0, target.energy - self.attack_damage)

class Apple:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)

class ScoreZone:
    def __init__(self, x, y): self.pos = np.array([x, y], dtype=np.float64)

# --- Data Logging ---
def setup_csv_logger(filename):
    header = ['generation', 'creatures_remaining', 'best_fitness', 'avg_fitness', 'total_apples_deposited']
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as f:
        csv.writer(f).writerow(header)

def log_to_csv(filename, stats):
    with open(filename, mode='a', newline='') as f:
        csv.DictWriter(f, fieldnames=stats.keys()).writerow(stats)

# --- NEW: Tournament Selection & Reproduction ---
def reproduction(parents, config):
    next_pop = []
    
    # Elitism
    parents.sort(key=lambda c: c.fitness, reverse=True)
    for i in range(min(config['elitism_count'], len(parents))):
        nn = LSTM_NeuralNetwork(Creature.NUM_INPUTS, config['hidden_size'], Creature.NUM_OUTPUTS)
        nn.set_weights(parents[i].nn.get_weights())
        next_pop.append(Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config))

    # Tournament Selection for remaining slots
    while len(next_pop) < config['population_size']:
        # Select two parents via tournaments
        p1_candidates = random.sample(parents, k=config['tournament_size'])
        p2_candidates = random.sample(parents, k=config['tournament_size'])
        p1 = max(p1_candidates, key=lambda c: c.fitness)
        p2 = max(p2_candidates, key=lambda c: c.fitness)

        # Crossover
        p1_w, p2_w = p1.nn.get_weights(), p2.nn.get_weights()
        cross_pt = random.randint(1, len(p1_w) - 1)
        child_w = np.concatenate([p1_w[:cross_pt], p2_w[cross_pt:]])
        
        # Create and mutate child
        nn = LSTM_NeuralNetwork(Creature.NUM_INPUTS, config['hidden_size'], Creature.NUM_OUTPUTS)
        nn.set_weights(child_w)
        nn.mutate(config['mutation_rate'], config['mutation_strength'])
        next_pop.append(Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), nn, config))

    return next_pop


# --- Headless Simulation Runner ---
def run_headless_simulation(config, total_generations, results_csv_path):
    cfg = {**DEFAULT_CONFIG, **config}
    setup_csv_logger(results_csv_path)

    food_zone_size = 120
    q_width, q_height = SCREEN_WIDTH / 4, SCREEN_HEIGHT / 4
    food_zones = [
        (q_width - food_zone_size/2, q_width + food_zone_size/2, q_height - food_zone_size/2, q_height + food_zone_size/2),
        (SCREEN_WIDTH-q_width - food_zone_size/2, SCREEN_WIDTH-q_width + food_zone_size/2, q_height-food_zone_size/2, q_height+food_zone_size/2),
        (q_width - food_zone_size/2, q_width + food_zone_size/2, SCREEN_HEIGHT-q_height-food_zone_size/2, SCREEN_HEIGHT-q_height+food_zone_size/2),
        (SCREEN_WIDTH-q_width - food_zone_size/2, SCREEN_WIDTH-q_width + food_zone_size/2, SCREEN_HEIGHT-q_height-food_zone_size/2, SCREEN_HEIGHT-q_height+food_zone_size/2)
    ]
    def spawn_apple(): return Apple(random.uniform(*random.choice(food_zones)[:2]), random.uniform(*random.choice(food_zones)[2:]))

    score_zone = ScoreZone(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
    apples = [spawn_apple() for _ in range(cfg['max_apples'])]
    population = [Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), LSTM_NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_size'], Creature.NUM_OUTPUTS), cfg) for _ in range(cfg['population_size'])]
    enemies = [Enemy(300, 300), Enemy(500, 400)]

    for gen in range(1, total_generations + 1):
        for frame in range(cfg['generation_time']):
            if not population: break
            if len(apples) < cfg['max_apples'] and random.random() < cfg['apple_respawn_rate']: apples.append(spawn_apple())
            for i in range(len(population) - 1, -1, -1):
                if not population[i].update(apples, score_zone, enemies, population): population.pop(i)
            for enemy in enemies: enemy.update(population)
        
        print(f"  Gen {gen}/{total_generations} complete for '{config.get('name', 'N/A')}'. Pop left: {len(population)}")
        
        if population:
            stats = {
                'generation': gen, 'creatures_remaining': len(population),
                'best_fitness': f"{max(c.fitness for c in population):.2f}",
                'avg_fitness': f"{sum(c.fitness for c in population) / len(population):.2f}",
                'total_apples_deposited': sum(c.apples_deposited_total for c in population)
            }
            log_to_csv(results_csv_path, stats)
            population = reproduction(population, cfg)
        else:
            log_to_csv(results_csv_path, {'generation': gen, 'creatures_remaining': 0, 'best_fitness': 0, 'avg_fitness': 0, 'total_apples_deposited': 0})
            population = [Creature(random.uniform(WORLD_PADDING, SCREEN_WIDTH - WORLD_PADDING), random.uniform(WORLD_PADDING, SCREEN_HEIGHT - WORLD_PADDING), LSTM_NeuralNetwork(Creature.NUM_INPUTS, cfg['hidden_size'], Creature.NUM_OUTPUTS), cfg) for _ in range(cfg['population_size'])]
        
        # Reset memory for the new generation
        for creature in population:
            creature.nn.reset_state()
