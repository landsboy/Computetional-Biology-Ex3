import numpy as np
import os

# A class that represents the network:
class Predictor:
    def __init__(self, architecture, weights, biases):
        self.weights_list = []
        self.biases_list = [] 
        for i in range(len(architecture)):
            self.weights_list.append(np.array(weights[i]).reshape(architecture[i][0], architecture[i][1]))
            self.biases_list.append(biases[i]) 

    def process(self, data_input):
        input_data = data_input
        for i, weights in enumerate(self.weights_list):
            output_data = np.dot(input_data, weights) + self.biases_list[i] 
            output_data = activation_sign(output_data)
            input_data = output_data
        return np.ravel(output_data)


# A function that puts the data we want to predict into the network:
def predict(network, output_file, test_data):
    with open(output_file, "w") as file:
        for sample in test_data:
            output = network.process(sample)
            if output >= 0:
                file.write(f"{int(output)}\n")
            else:
                file.write(f"{0}\n")
                
    print("The prediction ended successfully!\nGo watch in predic_results1.txt")
            

def activation_sign(input_data):
    return np.sign(input_data)

# A function that receives the wnet1 and parsing it:
def import_network(file_name):
    with open(file_name, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    weights = eval(lines[0].split(':')[1].strip())
    biases = eval(lines[1].split(':')[1].strip())
    model = eval(lines[2].split(':')[1].strip())

    architecture = [[model[0], model[1], "sign"]]
    return architecture, weights, biases

# The main function, receives 2 files - wnet1 and testnet1 file and a prediction will be made for the samples:
if __name__ == '__main__':
    # First we will check that the file wnet1 exists:
    if not os.path.exists("wnet1.txt"):
        print("Please take a look at the run instructions!")
        exit(1)

    input_file = input("Please enter the path to the sample file you want me to predict: ")

    # We will build the network from the data we have in wnet1
    architecture, weights, biases = import_network("wnet1.txt")
    network = Predictor(architecture, [weights], [biases])

    # We will create a test set from the input file we received:
    with open(input_file, 'r') as file:
        lines = file.readlines()
        test_data = [[int(num) for num in line.strip()] for line in lines]

    # We will make a prediction and record it in the file "predic_results1":
    predict(network, "predic_results1.txt" , test_data)