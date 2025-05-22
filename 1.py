    #Тоже более менее
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import networkx as nx

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

# Класс нейрона
class Neuron:
    def __init__(self, input_size, output_size):
        self.weights_in = np.random.randn(input_size) * 0.1 + np.random.normal(0, 0.2, input_size)  # Дополнительный шум
        self.weights_out = np.random.randn(output_size) * 0.1 + np.random.normal(0, 0.2, output_size)
        self.fitness = 0.0
        self.usage_count = 0

# Класс алгоритма SANE
class SANE:
    def __init__(self, pop_size, input_size, hidden_size, output_size):
        self.pop_size = pop_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size, output_size) for _ in range(pop_size)]
        self.network = NeuralNetwork(input_size, hidden_size, output_size)
        self.fitness_history = []
        self.best_network = None
        self.best_fitness = float('-inf')
        self.best_indices = None
        self.stagnation_count = 0
        self.STAGNATION_THRESHOLD = 50
        self.global_mutation_triggered = False
        self.best_solution_counter = 0
        self.epoch = 0
#Формирование сети
    def form_network(self, neuron_indices):
        weights_in = np.array([self.neurons[i].weights_in for i in neuron_indices]).T
        weights_out = np.array([self.neurons[i].weights_out for i in neuron_indices])
        self.network.set_weights(weights_in, weights_out)
        return self.network
#Ошибка или выборка
    def evaluate(self, env, max_steps=500):
        fitness_scores = []
        neuron_fitnesses = np.array([neuron.fitness for neuron in self.neurons])

        penalty = 0.6 * np.exp(self.epoch / 500)
        adjusted_fitnesses = neuron_fitnesses - penalty * np.array([n.usage_count for n in self.neurons])
        max_fitness = np.max(adjusted_fitnesses)
        if max_fitness - np.min(adjusted_fitnesses) > 1e-6:
            exp_fitnesses = np.exp(adjusted_fitnesses - max_fitness)
            selection_probs = exp_fitnesses / np.sum(exp_fitnesses)
        else:
            selection_probs = np.ones(self.pop_size) / self.pop_size

        for _ in range(80):
            if np.random.rand() < 0.3:
                indices = np.random.choice(self.pop_size, self.hidden_size, replace=False)
            else:
                tournament = np.random.choice(self.pop_size, 15, replace=False, p=selection_probs)
                indices = tournament[np.argmax([adjusted_fitnesses[i] for i in tournament])]
                indices = np.random.choice(self.pop_size, self.hidden_size, replace=False, p=selection_probs)

            network = self.form_network(indices)
            fitness = self.run_episode(env, network, max_steps)
            fitness_scores.append((fitness, indices))
            for idx in indices:
                self.neurons[idx].fitness += fitness / 80
                self.neurons[idx].usage_count += 1

        avg_fitness = np.mean([f[0] for f in fitness_scores])
        best_episode_fitness = max([f[0] for f in fitness_scores])
        best_episode_indices = fitness_scores[np.argmax([f[0] for f in fitness_scores])][1]

        if best_episode_fitness > self.best_fitness:
            self.stagnation_count = 0
            self.best_fitness = best_episode_fitness
            self.best_indices = best_episode_indices
            self.best_network = self.form_network(best_episode_indices)
            self.best_solution_counter += 1
            record_episode(self, filename=f"best_solution_{self.best_solution_counter}_fitness_{self.best_fitness:.2f}.gif", gif=True)
        else:
            self.stagnation_count += 1

        self.fitness_history.append(avg_fitness)
        self.epoch += 1
        return avg_fitness
#Запуск эпизода
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
#Мутация
    def mutate(self, mutation_rate=0.6, mutation_scale_base=0.8):
        mutation_scale = mutation_scale_base * (1 + self.epoch / 500)
        if self.stagnation_count > self.STAGNATION_THRESHOLD // 10:
            mutation_scale *= (1 + self.stagnation_count / self.STAGNATION_THRESHOLD * 3)

# Улучшенный локальный поиск для лучших
        sorted_neurons = sorted(self.neurons, key=lambda x: x.fitness, reverse=True)
        for neuron in sorted_neurons[:int(0.3 * self.pop_size)]:
            if np.random.rand() < 0.5:
                gradient_scale = 0.2
                neuron.weights_in += np.random.randn(self.input_size) * gradient_scale
                neuron.weights_out += np.random.randn(self.output_size) * gradient_scale
                neuron.weights_in = np.clip(neuron.weights_in, -1, 1)
                neuron.weights_out = np.clip(neuron.weights_out, -1, 1)

        for neuron in self.neurons:
            if np.random.rand() < mutation_rate:
                fitness_normalized = (neuron.fitness - self.best_fitness) / (self.best_fitness + 1e-6)
                adjusted_scale = mutation_scale * (1 + max(0, -fitness_normalized * 3))
                if neuron.fitness < self.best_fitness * 0.2 or np.random.rand() < 0.4:
                    neuron.weights_in = np.random.randn(self.input_size) * 0.1 + np.random.normal(0, 0.15, self.input_size)
                    neuron.weights_out = np.random.randn(self.output_size) * 0.1 + np.random.normal(0, 0.15, self.output_size)
                    neuron.fitness = 0.0
                    neuron.usage_count = 0
                else:
                    neuron.weights_in += np.random.randn(self.input_size) * adjusted_scale
                    neuron.weights_out += np.random.randn(self.output_size) * adjusted_scale
                neuron.weights_in = np.clip(neuron.weights_in, -1, 1)
                neuron.weights_out = np.clip(neuron.weights_out, -1, 1)

        if self.stagnation_count > self.STAGNATION_THRESHOLD // 10 and not self.global_mutation_triggered:
            self.global_mutation_triggered = True
            for neuron in self.neurons:
                neuron.weights_in += np.random.randn(self.input_size) * mutation_scale * 10
                neuron.weights_out += np.random.randn(self.output_size) * mutation_scale * 10
                neuron.weights_in = np.clip(neuron.weights_in, -1, 1)
                neuron.weights_out = np.clip(neuron.weights_out, -1, 1)
#Скрещивание
    def crossover(self):
        sorted_neurons = sorted(self.neurons, key=lambda x: x.fitness, reverse=True)
        elite_size = int(0.3 * self.pop_size)
        new_neurons = sorted_neurons[:elite_size].copy()

        if self.epoch % 5 == 0:
            replace_count = int(0.15 * self.pop_size)
            new_neurons.extend([Neuron(self.input_size, self.output_size) for _ in range(replace_count)])

        if self.stagnation_count > self.STAGNATION_THRESHOLD // 2:
            replace_count = int(0.9 * self.pop_size)
            new_neurons.extend([Neuron(self.input_size, self.output_size) for _ in range(replace_count)])

        while len(new_neurons) < self.pop_size:
            parent1, parent2 = np.random.choice(sorted_neurons[:self.pop_size//2], 2, replace=False)
            child = Neuron(self.input_size, self.output_size)
            crossover_point = np.random.randint(0, self.input_size)
            child.weights_in = np.concatenate((parent1.weights_in[:crossover_point], parent2.weights_in[crossover_point:]))
            crossover_point = np.random.randint(0, self.output_size)
            child.weights_out = np.concatenate((parent1.weights_out[:crossover_point], parent2.weights_out[crossover_point:]))
            new_neurons.append(child)
        self.neurons = new_neurons[:self.pop_size]
#Сохранение весов
    def save_weights(self, filename="weights.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump([(n.weights_in, n.weights_out) for n in self.neurons], f)
#Загрузка весов
    def load_weights(self, filename="weights.pkl"):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        for neuron, (w_in, w_out) in zip(self.neurons, weights):
            neuron.weights_in = w_in
            neuron.weights_out = w_out

# Функции визуализации
def record_episode(sane, filename="lunar_lander.mp4", gif=False, use_best_network=True):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    frames = []
    if use_best_network and sane.best_network is not None:
        network = sane.best_network
    else:
        sorted_neurons = sorted(sane.neurons, key=lambda x: x.fitness, reverse=True)
        best_indices = [i for i, _ in enumerate(sorted_neurons[:sane.hidden_size])]
        network = sane.form_network(best_indices)
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
            G.add_edge(f"I{i}", f"H{j}")
    for i in range(network.hidden_size):
        for j in range(network.output_size):
            G.add_edge(f"H{i}", f"O{j}")
    pos = nx.multipartite_layout(G, subset_key="layer")
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10)
    plt.savefig(filename)
    plt.close()

# Главный цикл
def main():
    env = gym.make("LunarLander-v3")
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_size = 200
    pop_size = 1000
    epochs = 1500

    sane = SANE(pop_size, input_size, hidden_size, output_size)
    plot_network(sane.network, "network_initial.png")

    for epoch in range(epochs):
        fitness = sane.evaluate(env)
        print(f"Эпоха {epoch+1}, Средняя приспособленность: {fitness:.2f}, Лучшая: {sane.best_fitness:.2f}")

        if sane.stagnation_count >= sane.STAGNATION_THRESHOLD:
            print(f"Стагнация обнаружена на эпохе {epoch+1}, остановка.")
            break

        sane.mutate(mutation_rate=0.6, mutation_scale_base=0.8)
        sane.crossover()
        if epoch % 10 == 0:
            sane.save_weights(f"weights_epoch_{epoch}.pkl")
        if epoch == epochs // 2:
            plot_network(sane.network, "network_middle.png")
        if (epoch + 1) % 100 == 0:
            record_episode(sane, filename=f"landing_epoch_{epoch+1}.gif", gif=True)

    sane.save_weights("final_weights.pkl")
    plot_network(sane.network, "network_final.png")
    plot_fitness(sane.fitness_history)
    record_episode(sane, filename="lunar_lander.mp4", gif=False)

    env.close()

if __name__ == "__main__":
    main()