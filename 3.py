import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import networkx as nx
from sklearn.decomposition import PCA

# Класс нейронной сети
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_in = np.zeros((input_size, hidden_size))
        self.weights_out = np.zeros((hidden_size, output_size))

    def set_weights(self, weights_in, weights_out):
        self.weights_in = weights_in
        self.weights_out = weights_out

    def forward(self, inputs):
        hidden = np.tanh(np.dot(inputs, self.weights_in))
        output = np.dot(hidden, self.weights_out)
        return output

# Класс нейрона с бинарным кодированием весов
class Neuron:
    def __init__(self, input_size, output_size, bits=16):
        self.bits = bits
        self.weights_in = np.random.randint(0, 2**bits, input_size)  # Бинарное представление
        self.weights_out = np.random.randint(0, 2**bits, output_size)
        self.fitness = 0.0
        self.usage_count = 0

    def decode_weight(self, binary):
        # Преобразуем бинарное представление в [-1, 1]
        max_val = 2**self.bits - 1
        return 2 * (binary / max_val) - 1

# Класс алгоритма SANE
class SANE:
    def __init__(self, pop_size, input_size, hidden_size, output_size, blueprint_size=50):
        self.pop_size = pop_size
        self.blueprint_size = blueprint_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size, output_size) for _ in range(pop_size)]
        self.blueprints = [np.random.choice(pop_size, hidden_size, replace=False) for _ in range(blueprint_size)]
        self.blueprint_fitness = [0.0] * blueprint_size
        self.network = NeuralNetwork(input_size, hidden_size, output_size)
        self.fitness_history = []
        self.best_network = None
        self.best_fitness = float('-inf')
        self.best_indices = None
        self.stagnation_count = 0
        self.STAGNATION_THRESHOLD = 300
        self.epoch = 0
        self.best_solution_counter = 0

    def form_network(self, neuron_indices):
        # Декодируем веса из бинарного представления
        weights_in = np.array([self.neurons[i].decode_weight(self.neurons[i].weights_in) for i in neuron_indices]).T
        weights_out = np.array([self.neurons[i].decode_weight(self.neurons[i].weights_out) for i in neuron_indices])
        self.network.set_weights(weights_in, weights_out)
        return self.network

    def evaluate(self, env, max_steps=500):
        # Сбрасываем приспособленности
        for neuron in self.neurons:
            neuron.fitness = 0.0
            neuron.usage_count = 0
        for i, blueprint in enumerate(self.blueprints):
            network = self.form_network(blueprint)
            fitness = self.run_episode(env, network, max_steps)
            self.blueprint_fitness[i] = fitness
            for idx in blueprint:
                self.neurons[idx].usage_count += 1

        # Обновляем приспособленность нейронов (средняя по 5 лучшим ИНС)
        for idx, neuron in enumerate(self.neurons):
            if neuron.usage_count > 0:
                neuron_fitnesses = [
                    self.blueprint_fitness[i] for i, blueprint in enumerate(self.blueprints) if idx in blueprint
                ]
                neuron_fitnesses = sorted(neuron_fitnesses, reverse=True)[:5]
                neuron.fitness = np.mean(neuron_fitnesses) if neuron_fitnesses else 0.0

        # Сохраняем лучшую сеть
        best_idx = np.argmax(self.blueprint_fitness)
        if self.blueprint_fitness[best_idx] > self.best_fitness:
            self.best_fitness = self.blueprint_fitness[best_idx]
            self.best_indices = self.blueprints[best_idx]
            self.best_network = self.form_network(self.best_indices)
            self.stagnation_count = 0
            self.best_solution_counter += 1
            record_episode(self,
                           filename=f"best_solution_{self.best_solution_counter}_fitness_{self.best_fitness:.2f}.gif",
                           gif=True)
        else:
            self.stagnation_count += 1

        self.fitness_history.append(np.mean(self.blueprint_fitness))
        self.epoch += 1
        return np.mean(self.blueprint_fitness)

    def run_episode(self, env, network, max_steps):
        observation, _ = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action_probs = network.forward(observation)
            action_probs = np.clip(action_probs, -10, 10)
            exp_probs = np.exp(action_probs - np.max(action_probs))
            action_probs = exp_probs / np.sum(exp_probs)
            action = np.random.choice(np.arange(self.output_size), p=action_probs)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def mutate(self):
        for neuron in self.neurons:
            for i in range(len(neuron.weights_in)):
                for bit in range(neuron.bits):
                    if np.random.rand() < 0.001:  # 0.1% на бит
                        neuron.weights_in[i] ^= (1 << bit)
            for i in range(len(neuron.weights_out)):
                for bit in range(neuron.bits):
                    if np.random.rand() < 0.001:
                        neuron.weights_out[i] ^= (1 << bit)

    def crossover(self):
        sorted_neurons = sorted(self.neurons, key=lambda x: x.fitness, reverse=True)
        elite_size = int(0.25 * self.pop_size)  # 25% лучших
        new_neurons = sorted_neurons[:elite_size].copy()

        while len(new_neurons) < self.pop_size:
            parent1, parent2 = np.random.choice(sorted_neurons[:self.pop_size//2], 2, replace=False)
            child = Neuron(self.input_size, self.output_size)
            crossover_point = np.random.randint(0, self.input_size)
            child.weights_in = np.concatenate((parent1.weights_in[:crossover_point], parent2.weights_in[crossover_point:]))
            crossover_point = np.random.randint(0, self.output_size)
            child.weights_out = np.concatenate((parent1.weights_out[:crossover_point], parent2.weights_out[crossover_point:]))
            new_neurons.append(child)
        self.neurons = new_neurons[:self.pop_size]

    def crossover_blueprints(self):
        sorted_indices = np.argsort(self.blueprint_fitness)[::-1]
        elite_size = int(0.25 * self.blueprint_size)
        new_blueprints = [self.blueprints[i] for i in sorted_indices[:elite_size]]

        while len(new_blueprints) < self.blueprint_size:
            parent1, parent2 = np.random.choice(sorted_indices[:self.blueprint_size//2], 2, replace=False)
            crossover_point = np.random.randint(1, self.hidden_size)
            child = np.concatenate((self.blueprints[parent1][:crossover_point], self.blueprints[parent2][crossover_point:]))
            new_blueprints.append(child)
        self.blueprints = new_blueprints[:self.blueprint_size]

    def mutate_blueprints(self):
        for blueprint in self.blueprints:
            for i in range(self.hidden_size):
                if np.random.rand() < 0.01:  # 1% вероятность
                    new_idx = np.random.randint(self.pop_size)
                    blueprint[i] = new_idx
                elif np.random.rand() < 0.5:  # 50% вероятность выбрать нейрон-потомок
                    new_idx = np.random.randint(self.pop_size//2, self.pop_size)
                    blueprint[i] = new_idx

    def save_weights(self, filename="weights.pkl"):
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'neurons': [(n.weights_in, n.weights_out) for n in self.neurons],
            'best_indices': self.best_indices
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_weights(self, filename="weights.pkl"):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.input_size = data['input_size']
        self.hidden_size = data['hidden_size']
        self.output_size = data['output_size']
        self.neurons = [Neuron(self.input_size, self.output_size) for _ in range(self.pop_size)]
        for neuron, (w_in, w_out) in zip(self.neurons, data['neurons']):
            neuron.weights_in = w_in
            neuron.weights_out = w_out
        self.best_indices = data['best_indices']
        self.best_network = self.form_network(self.best_indices)

# Функции визуализации
def record_episode(sane, filename="lunar_lander.mp4", gif=False, use_best_network=True):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []
    if use_best_network and sane.best_network is not None:
        network = sane.best_network
    else:
        sorted_indices = np.argsort(sane.blueprint_fitness)[::-1]
        network = sane.form_network(sane.blueprints[sorted_indices[0]])
    observation, _ = env.reset()
    for _ in range(500):
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
            weight = network.weights_in[i, j]
            G.add_edge(f"I{i}", f"H{j}", weight=abs(weight))
    for i in range(network.hidden_size):
        for j in range(network.output_size):
            weight = network.weights_out[i, j]
            G.add_edge(f"H{i}", f"O{j}", weight=abs(weight))
    pos = nx.multipartite_layout(G, subset_key="layer")
    weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, width=weights)
    plt.savefig(filename)
    plt.close()

def plot_weight_diversity(sane, filename="weight_diversity.png"):
    weights = np.array([n.decode_weight(n.weights_in) for n in sane.neurons])
    pca = PCA(n_components=2)
    projections = pca.fit_transform(weights)
    plt.scatter(projections[:, 0], projections[:, 1], c='blue', alpha=0.5)
    plt.xlabel("Первая главная компонента")
    plt.ylabel("Вторая главная компонента")
    plt.title("Разнообразие весов нейронов")
    plt.savefig(filename)
    plt.close()

# Главный цикл
def main():
    env = gym.make("LunarLander-v3")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 20
    pop_size = 6000
    blueprint_size = 100
    epochs = 1500

    sane = SANE(pop_size, input_size, hidden_size, output_size, blueprint_size)
    plot_network(sane.network, "network_initial.png")
    plot_weight_diversity(sane, "weight_diversity_initial.png")

    for epoch in range(epochs):
        fitness = sane.evaluate(env)
        print(f"Эпоха {epoch+1}, Средняя приспособленность: {fitness:.2f}, Лучшая: {sane.best_fitness:.2f}")

        if sane.stagnation_count >= sane.STAGNATION_THRESHOLD:
            print(f"Стагнация обнаружена на эпохе {epoch+1}, остановка.")
            break

        sane.mutate()
        sane.crossover()
        sane.crossover_blueprints()
        sane.mutate_blueprints()

        if epoch % 10 == 0:
            sane.save_weights(f"weights_epoch_{epoch}.pkl")
        if epoch == epochs // 2:
            plot_network(sane.network, "network_middle.png")
            plot_weight_diversity(sane, "weight_diversity_middle.png")
        if (epoch + 1) % 100 == 0:
            record_episode(sane, filename=f"landing_epoch_{epoch+1}.gif", gif=True)

    sane.save_weights("final_weights.pkl")
    plot_network(sane.network, "network_final.png")
    plot_weight_diversity(sane, "weight_diversity_final.png")
    plot_fitness(sane.fitness_history)
    record_episode(sane, filename="lunar_lander.mp4", gif=False)

    env.close()

if __name__ == "__main__":
    main()