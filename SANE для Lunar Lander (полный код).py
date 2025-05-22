import numpy as np
import gymnasium as gym
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import imageio
import networkx as nx
import pickle
from multiprocessing import Pool

# Класс нейрона с ограниченными связями
class Neuron:
    def __init__(self, input_size, output_size, num_connections=3):
        self.label = np.random.randint(0, 256)  # 8-битная метка (lec8.pdf, рис. 8.1)
        self.num_connections = num_connections
        self.input_indices = np.random.choice(input_size, num_connections, replace=False)
        self.output_indices = np.random.choice(output_size, num_connections, replace=False)
        self.weights_in = np.random.randn(num_connections) * 1.0  # Увеличено с 0.1 до 1.0
        self.weights_out = np.random.randn(num_connections) * 1.0  # Для разнообразия действий
        self.fitness = 0.0
        self.usage_count = 0

# Класс blueprint (комбинация нейронов)
class Blueprint:
    def __init__(self, hidden_size, pop_size):
        self.neuron_indices = np.random.choice(pop_size, hidden_size, replace=False)
        self.fitness = 0.0

# Класс нейронной сети прямого распространения
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_in = np.zeros((input_size, hidden_size))
        self.weights_out = np.zeros((hidden_size, output_size))

    def set_weights(self, neurons, neuron_indices):
        self.weights_in = np.zeros((self.input_size, self.hidden_size))
        self.weights_out = np.zeros((self.hidden_size, self.output_size))
        for i, idx in enumerate(neuron_indices):
            neuron = neurons[idx]
            for j, input_idx in enumerate(neuron.input_indices):
                self.weights_in[input_idx, i] = neuron.weights_in[j]
            for j, output_idx in enumerate(neuron.output_indices):
                self.weights_out[i, output_idx] = neuron.weights_out[j]

    def forward(self, inputs):
        hidden = np.tanh(np.dot(inputs, self.weights_in))
        output = np.dot(hidden, self.weights_out)
        return output

# Класс SANE
class SANE:
    def __init__(self, pop_size, input_size, hidden_size, output_size, blueprint_size=20):  # 20 для теста
        self.pop_size = pop_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.blueprint_size = blueprint_size
        self.neurons = [Neuron(input_size, output_size, num_connections=3) for _ in range(pop_size)]
        self.blueprints = [Blueprint(hidden_size, pop_size) for _ in range(blueprint_size)]
        self.network = NeuralNetwork(input_size, hidden_size, output_size)
        self.fitness_history = []
        self.best_network = None
        self.best_fitness = float('-inf')
        self.best_indices = None
        self.epoch = 0
        self.stagnation_count = 0
        self.STAGNATION_THRESHOLD = 50
        self.network_cache = {}  # Кэш для оптимизации

    def form_network(self, neuron_indices):
        key = tuple(neuron_indices)
        if key not in self.network_cache:
            network = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            network.set_weights(self.neurons, neuron_indices)
            self.network_cache[key] = network
        return self.network_cache[key]

    def run_episode(self, env, network, max_steps=1000):  # Увеличено до 1000
        observation, _ = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action_probs = network.forward(observation)
            action_probs = np.clip(action_probs, -10, 10)
            exp_probs = np.exp(action_probs - np.max(action_probs))
            action_probs = exp_probs / np.sum(exp_probs)
            action = np.random.choice(np.arange(self.output_size), p=action_probs)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if self.epoch % 10 == 0:  # Отладка каждые 10 эпох
                print(f"Epoch {self.epoch}, Step {step}, Action: {action}, Prob: {action_probs}, Reward: {reward}, Total: {total_reward}")
            if terminated or truncated:
                break
        return total_reward

    def evaluate_network(self, args):
        blueprint, env, max_steps = args
        network = self.form_network(blueprint.neuron_indices)
        fitness = self.run_episode(env, network, max_steps)
        return fitness, blueprint

    def evaluate(self, env, max_steps=1000):
        for neuron in self.neurons:
            neuron.fitness = 0.0
            neuron.usage_count = 0

        with Pool() as pool:
            results = pool.map(self.evaluate_network, [(b, env, max_steps) for b in self.blueprints])

        for fitness, blueprint in results:
            blueprint.fitness = fitness
            for idx in blueprint.neuron_indices:
                self.neurons[idx].usage_count += 1

        for neuron in self.neurons:
            if neuron.usage_count > 0:
                blueprint_fitnesses = [b.fitness for b in self.blueprints if neuron in [self.neurons[i] for i in b.neuron_indices]]
                top_fitnesses = sorted(blueprint_fitnesses, reverse=True)[:5]
                neuron.fitness = np.mean(top_fitnesses) if top_fitnesses else 0.0

        avg_fitness = np.mean([max(b.fitness, -100) for b in self.blueprints])  # Ограничим минимум -100
        best_fitness = max([b.fitness for b in self.blueprints])
        best_blueprint = self.blueprints[np.argmax([b.fitness for b in self.blueprints])]

        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_indices = best_blueprint.neuron_indices
            self.best_network = self.form_network(best_blueprint.neuron_indices)
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if self.stagnation_count > self.STAGNATION_THRESHOLD:
            self.neurons[int(0.5 * self.pop_size):] = [Neuron(self.input_size, self.output_size, num_connections=3) for _ in range(int(0.5 * self.pop_size))]
            self.stagnation_count = 0

        self.fitness_history.append(avg_fitness)
        self.epoch += 1
        return avg_fitness

    def crossover_neurons(self):
        sorted_neurons = sorted(self.neurons, key=lambda x: x.fitness, reverse=True)
        elite_size = int(0.25 * self.pop_size)
        new_neurons = sorted_neurons[:elite_size].copy()

        while len(new_neurons) < self.pop_size:
            parent1, parent2 = np.random.choice(sorted_neurons[:self.pop_size//2], 2, replace=False)
            child = Neuron(self.input_size, self.output_size, num_connections=3)
            crossover_point = np.random.randint(0, 3)
            child.input_indices = np.concatenate((parent1.input_indices[:crossover_point], parent2.input_indices[crossover_point:]))
            child.output_indices = np.concatenate((parent1.output_indices[:crossover_point], parent2.output_indices[crossover_point:]))
            child.weights_in = np.concatenate((parent1.weights_in[:crossover_point], parent2.weights_in[crossover_point:]))
            child.weights_out = np.concatenate((parent1.weights_out[:crossover_point], parent2.weights_out[crossover_point:]))
            new_neurons.append(child)
            new_neurons.append(parent1)  # Сохраняем родителя
        self.neurons = new_neurons[:self.pop_size]

    def mutate_neurons(self):
        for neuron in self.neurons:
            if np.random.rand() < 0.01:  # Увеличено с 0.001
                neuron.weights_in += np.random.randn(neuron.num_connections) * 0.1
                neuron.weights_out += np.random.randn(neuron.num_connections) * 0.1
                neuron.weights_in = np.clip(neuron.weights_in, -1, 1)
                neuron.weights_out = np.clip(neuron.weights_out, -1, 1)

    def crossover_blueprints(self):
        sorted_blueprints = sorted(self.blueprints, key=lambda x: x.fitness, reverse=True)
        elite_size = int(0.25 * self.blueprint_size)
        new_blueprints = sorted_blueprints[:elite_size].copy()

        while len(new_blueprints) < self.blueprint_size:
            parent1, parent2 = np.random.choice(sorted_blueprints[:self.blueprint_size//2], 2, replace=False)
            child = Blueprint(self.hidden_size, self.pop_size)
            crossover_point = np.random.randint(0, self.hidden_size)
            child.neuron_indices = np.concatenate((parent1.neuron_indices[:crossover_point], parent2.neuron_indices[crossover_point:]))
            new_blueprints.append(child)
        self.blueprints = new_blueprints[:self.blueprint_size]

    def mutate_blueprints(self):
        for blueprint in self.blueprints[1:]:  # Пропуск элиты
            if np.random.rand() < 0.1:  # Увеличено с 0.01
                idx = np.random.randint(0, self.hidden_size)
                new_neuron = np.random.randint(0, self.pop_size)
                if np.random.rand() < 0.5:
                    sorted_neurons = sorted(self.neurons, key=lambda x: x.fitness, reverse=True)
                    new_neuron = np.random.choice([i for i, _ in enumerate(sorted_neurons[:self.pop_size//2])])
                blueprint.neuron_indices[idx] = new_neuron

    def save_weights(self, filename="weights.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump([(n.input_indices, n.output_indices, n.weights_in, n.weights_out) for n in self.neurons], f)

    def load_weights(self, filename="weights.pkl"):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        for neuron, (in_idx, out_idx, w_in, w_out) in zip(self.neurons, weights):
            neuron.input_indices = in_idx
            neuron.output_indices = out_idx
            neuron.weights_in = w_in
            neuron.weights_out = w_out

    def analyze_weights(self, filename="weights_pca.png"):
        weights = np.array([n.weights_in for n in self.neurons])
        pca = PCA(n_components=2)
        projections = pca.fit_transform(weights)
        plt.scatter(projections[:, 0], projections[:, 1], c='blue', alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA весов нейронов (эпоха {self.epoch})")
        plt.savefig(filename)
        plt.close()

    def analyze_neuron_roles(self, env, filename="neuron_roles.txt"):
        roles = []
        for i in range(self.hidden_size):
            temp_network = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
            temp_network.set_weights(self.neurons, self.best_indices)
            for j in range(self.hidden_size):
                if j != i:
                    temp_network.weights_in[:, j] = 0
                    temp_network.weights_out[j, :] = 0
            observation, _ = env.reset()
            action_probs = temp_network.forward(observation)
            roles.append(f"Нейрон {i}: Влияет на действия {action_probs}")
        with open(filename, 'w') as f:
            for role in roles:
                f.write(role + '\n')

# Функции визуализации
def record_episode(sane, filename="lunar_lander.mp4", gif=False, use_best_network=True):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []
    network = sane.best_network if use_best_network and sane.best_network is not None else sane.form_network(sane.blueprints[0].neuron_indices)
    observation, _ = env.reset()
    for _ in range(1000):  # Синхронизировано с max_steps
        frame = env.render()
        frames.append(frame)
        action_probs = network.forward(observation)
        exp_probs = np.exp(action_probs - np.max(action_probs))
        action_probs = exp_probs / np.sum(exp_probs)
        action = np.random.choice(np.arange(sane.output_size), p=action_probs)
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()
    if gif:
        with imageio.get_writer(filename, mode='I', fps=30) as writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        with imageio.get_writer(filename, fps=30, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)

def plot_fitness(fitness_history, filename="convergence.png"):
    plt.plot(fitness_history)
    plt.xlabel("Эпохи")
    plt.ylabel("Средняя приспособленность")
    plt.title("Сходимость SANE")
    plt.savefig(filename)
    plt.close()

def plot_network(network, filename="network.png"):
    G = nx.DiGraph()
    for i in range(network.input_size):
        G.add_node(f"I{i}", layer="input")
    for i in range(network.hidden_size):
        G.add_node(f"H{i}", layer="hidden")
    for i in range(network.output_size):
        G.add_node(f"O{i}", layer="output")
    for i in range(network.input_size):
        for j in range(network.hidden_size):
            G.add_edge(f"I{i}", f"H{j}")
    for i in range(network.hidden_size):
        for j in range(network.output_size):
            G.add_edge(f"H{i}", f"O{j}")
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.savefig(filename)
    plt.close()

# Основной цикл
def main():
    env = gym.make("LunarLander-v3")
    input_size = env.observation_space.shape[0]  # 8
    output_size = env.action_space.n  # 4
    hidden_size = 8
    pop_size = 100  # Уменьшено для теста (оригинал: 400)
    blueprint_size = 20  # Уменьшено для теста (оригинал: 80)
    epochs = 100  # Для теста (оригинал: 1500)

    sane = SANE(pop_size, input_size, hidden_size, output_size, blueprint_size)
    plot_network(sane.network, "network_initial.png")

    for epoch in range(epochs):
        fitness = sane.evaluate(env)
        print(f"Эпоха {epoch+1}, Средняя приспособленность: {fitness:.2f}, Лучшая: {sane.best_fitness:.2f}")
        sane.crossover_neurons()
        sane.mutate_neurons()
        sane.crossover_blueprints()
        sane.mutate_blueprints()
        if epoch % 10 == 0:
            sane.save_weights(f"weights_epoch_{epoch}.pkl")
            sane.analyze_weights(f"weights_pca_epoch_{epoch}.png")
            sane.analyze_neuron_roles(env, f"neuron_roles_epoch_{epoch}.txt")
        if epoch == epochs // 2:
            plot_network(sane.network, "network_middle.png")
        if (epoch + 1) % 100 == 0:
            record_episode(sane, filename=f"landing_epoch_{epoch+1}.gif", gif=True)
        if sane.stagnation_count >= sane.STAGNATION_THRESHOLD:
            print(f"Стагнация обнаружена на эпохе {epoch+1}, остановка.")
            break

    sane.save_weights("final_weights.pkl")
    plot_network(sane.network, "network_final.png")
    plot_fitness(sane.fitness_history)
    record_episode(sane, filename="lunar_lander.mp4", gif=False)
    env.close()

if __name__ == "__main__":
    main()

