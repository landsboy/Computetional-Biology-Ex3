from sklearn.model_selection import train_test_split
import copy
import random
import numpy as np

# Global variebles:
pop_size = 100
GENE = 500
sel_rate = 0.5
mut_rate = 0.01
MINIMA = 20

# A class representing a neural network:
class NeuralNet:
    def __init__(self, mod):
        self.biases = []
        self.weights = []

        for layer in mod:
            self.weights.append(np.random.randn(layer[0], layer[1]) * np.sqrt(1 / layer[0]))
            self.biases.append(np.random.randn(layer[1]))

    def propagate_f(self, data):
        z = None
        for i, weights in enumerate(self.weights):
            z = np.dot(data, weights) + self.biases[i]
            z = np.sign(z)
        return np.ravel(z)

# A class that represents a single entity from the entire population:
class Entity:
    X = None  
    y = None  

    def __init__(self, model):
        self.neural_network = NeuralNet(model)
        self.fitness = 0

    def fitness_calc(self):
        predictions = self.neural_network.propagate_f(Entity.X)
        self.fitness = round(float(np.mean((predictions > 0).astype(int) == Entity.y)), 4)

# A method that receives the sample file and divides them into train and test files:
def load_f(file_path, test_size=0.2):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        for line in lines:
            binary_str, label = line.strip().split()
            binary_data = [int(bit) for bit in binary_str]
            data.append(binary_data + [int(label)])
    features = [sample[:-1] for sample in data]
    labels = [sample[-1] for sample in data]
    X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# A helper function for the genetic algorithm - performs a crossover in the population
def crossover_f(pop, network):
    offspring = []
    for _ in range((len(pop))):
        parent1, parent2 = random.sample(pop, 2)
        offspring_entity = Entity(network)
        offspring_weights = []
        offspring_biases = []
        for weights1, weights2, biases1, biases2 in zip(parent1.neural_network.weights, parent2.neural_network.weights, parent1.neural_network.biases, parent2.neural_network.biases):
            genes1 = weights1.flatten()
            genes2 = weights2.flatten()
            split = random.randint(0, len(genes1) - 1)
            offspring_genes = np.concatenate((genes1[:split], genes2[split:]))
            
            offspring_weights.append(offspring_genes.reshape(weights1.shape))
            bias_genes1 = biases1.flatten()
            bias_genes2 = biases2.flatten()
            split = random.randint(0, len(bias_genes1) - 1)
            
            offspring_bias_genes = np.concatenate((bias_genes1[:split], bias_genes2[split:]))
            offspring_biases.append(offspring_bias_genes.reshape(biases1.shape))
        offspring_entity.neural_network.weights = offspring_weights
        offspring_entity.neural_network.biases = offspring_biases
        offspring.append(offspring_entity)
    return offspring

# A helper function for the genetic algorithm - creats nutation in the population
def mutation_f(pop):
    for entity in pop:
        for i in range(len(entity.neural_network.weights)):
            layer_weights = entity.neural_network.weights[i]
            layer_biases = entity.neural_network.biases[i]

            weight_mask = np.random.choice([True, False], size=layer_weights.shape, p=[mut_rate, 1 - mut_rate])
            new_weights = np.random.normal(0, 0.1, size=weight_mask.sum())
            layer_weights[weight_mask] = new_weights
            
            bias_mask = np.random.choice([True, False], size=layer_biases.shape, p=[mut_rate, 1 - mut_rate])
            new_biases = np.random.normal(0, 0.1, size=bias_mask.sum())
            layer_biases[bias_mask] = new_biases
    return pop

# The main function of the GA - receiving a train test and using a genetic algorithm to find the most accurate weights and biases:
def run_ga(x, y, model):
    Iter = []
    Entity.X = x 
    Entity.y = y 
    pop = [Entity(model) for _ in range(pop_size)]
    
    for entity in pop:
        entity.fitness_calc()

    max_fitness_prev =  -np.inf
    pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
    max_fitness_prev = max(max_fitness_prev, pop[0].fitness)

    local_minima = 0
    fitness_list = []
    for gens_num in range(GENE):
        print(f"Generation : {gens_num}")
        Iter.append(gens_num)
        fitness_scores = [entity.fitness for entity in pop]
        max_fitness = np.max(fitness_scores)
        fitness_list.append(max_fitness)
        for entity in pop:
            new_entity = copy.deepcopy(entity)
            new_entity = mutation_f([new_entity])[0]
            new_entity.fitness_calc()
            
            if new_entity.fitness > entity.fitness:
                entity.neural_network = new_entity.neural_network
                entity.fitness = new_entity.fitness
        
        sorted_pop = sorted(pop, key=lambda entity: entity.fitness, reverse=True)
        best = sorted_pop[:int(sel_rate * len(sorted_pop))]
        offspring = crossover_f(best, model)
        mutated_offspring = mutation_f(offspring)
        
        for entity in mutated_offspring:
            entity.fitness_calc()
        
        pop.extend(mutated_offspring)
        pop = sorted(pop, key=lambda x: x.fitness, reverse=True)
        pop = pop[:pop_size]
        max_fitness = pop[0].fitness
        
        if max_fitness <= max_fitness_prev:
            local_minima += 1
            if local_minima >= MINIMA:
                break
        else:
            local_minima = 0
            max_fitness_prev = max_fitness
        gens_num += 1
    return pop[0]
    
# The main function receives the learning file and the test file and runs the genetic algorithm using a neural network:
if __name__ == '__main__':
    input_file = input("Enter the PATH to all sampels file (I will split this to train&test): ")

    # First we will split the file nn1.txt into two sets - train and test:
    X_train, X_test, y_train, y_test = load_f(input_file)

    # The neural network we chose - the perceptron:
    perceptron = [[16, 1]]

    # Running a GA to find the best weights and bias:
    best = run_ga(X_train, y_train, perceptron)

    # Prediction of the trained model on the test set, and checking its accuracy:
    test_predictions = best.neural_network.propagate_f(X_test)
    acc = np.mean((test_predictions > 0).astype(int) == y_test)
    print("Test Set Accuracy:", acc)

    #save the results:
    with open("wnet1.txt", 'w') as file:
        file.write(f"weights: {[layer.tolist() for layer in best.neural_network.weights][0]} \nbiases: {[layer.tolist() for layer in best.neural_network.biases][0][0]}\nmodel: {perceptron[0]}")
    print("\nDONE!\nGo watch in wnet1.txt")