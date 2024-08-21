import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from CNN_model import Conv1DNet
import torch.nn as nn

# Adjustable Parameters
# ----------------------
input_size = 5  # Number of input features
output_size = 1  # Number of output features (target)
learning_rate = 0.0001  # Learning rate for the optimizer
num_epochs = 20000  # Number of epochs to train the model
batch_size = 25  # Batch size for training and testing
channels_1 = 32  # Number of channels in the first convolutional layer
channels_2 = 64  # Number of channels in the second convolutional layer
n_fish = 10  # Number of fish in the swarm
n_iter = 50  # Number of iterations for the fish swarm optimization
visual = 5  # Visual range for fish in the swarm
step = 1  # Step size for fish movement
delta = 0.1  # Congestion factor in the swarm
# ----------------------

# Load data from Excel file
excel_file = "./image.xlsx"
data_frame = pd.read_excel(excel_file).to_numpy()

# Split data into training and testing sets
X_train = data_frame[1:26, 1:6].astype(float)
y_train = data_frame[1:26, 6].astype(float)
y_train = y_train[:, np.newaxis]

X_test = data_frame[27:, 1:6].astype(float)
y_test = data_frame[27:, 6].astype(float)
y_test = y_test[:, np.newaxis]

# Convert data to torch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create dataset and data loaders
train_dataset = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(X_test, y_test)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
net = Conv1DNet(input_size, output_size, channels_1, channels_2)

# Define a hook function for capturing activations
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Define a function to compute loss
def get_loss(input_size, output_size, x):
    torch.manual_seed(99)  # Set the random seed for reproducibility
    fish_x, fish_y = x

    # Instantiate a new model with parameters from fish swarm
    net = Conv1DNet(input_size, output_size, fish_x, fish_y)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Training phase
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)

    # Evaluation phase
    net.eval()
    test_loss = 0.0
    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = net.compute_loss_t(outputs, labels)
        test_loss += loss.item()
    test_loss = test_loss / len(test_loader)

    return test_loss

# Define the Fish class, representing individual particles in the swarm
class Fish:
    def __init__(self, x):
        self.position = x  # Position in parameter space
        self.fitness = get_loss(input_size, output_size, x)  # Fitness score based on position

# Define the FishSwarm class, implementing the fish swarm optimization algorithm
class FishSwarm:
    def __init__(self, n_fish, n_iter, visual, step, delta):
        self.n_fish = n_fish  # Number of fish in the swarm
        self.n_iter = n_iter  # Number of iterations to run
        self.visual = visual  # Visual range of the fish
        self.step = step  # Step size for movement
        self.delta = delta  # Congestion factor
        self.fishes = []  # List to store fish objects
        self.best_fish = None  # The fish with the best fitness
        self.best_fitness = np.inf  # The best fitness score
        self.best_position = None  # The best position found
        self.fitness_curve = []  # List to store the fitness over iterations

        # Initialize the fish swarm
        for i in range(n_fish):
            x = np.random.randint(5, 128, 2)
            fish = Fish(x)
            self.fishes.append(fish)
            if fish.fitness < self.best_fitness:
                self.best_fish = fish
                self.best_fitness = fish.fitness
                self.best_position = fish.position

    # Define the forage behavior for the fish
    def forage(self, input_size, output_size, fish):
        new_position = fish.position + np.random.randint(-1, 1, 2) * self.step
        new_position = np.round(new_position).astype(int)
        for i in range(len(new_position)):
            if new_position[i] <= 0:
                new_position[i] += 5
        new_fitness = get_loss(input_size, output_size, new_position)
        if new_fitness < fish.fitness:
            fish.position = new_position
            fish.fitness = new_fitness
            if new_fitness < self.best_fitness:
                self.best_fish = fish
                self.best_fitness = new_fitness
                self.best_position = new_position

    # Define the swarm behavior
    def swarm(self, input_size, output_size, fish):
        n_local = 0
        center = np.zeros(2)
        mean_fitness = 0
        for other in self.fishes:
            if np.linalg.norm(fish.position - other.position) < self.visual:
                n_local += 1
                center += other.position
                mean_fitness += other.fitness
        if n_local > 0:
            center /= n_local
            mean_fitness /= n_local
            if mean_fitness < fish.fitness:
                new_position = fish.position + (center - fish.position) * np.random.randint(0, 1) * self.step / np.linalg.norm(center - fish.position)
                new_position = np.round(new_position).astype(int)
                for i in range(len(new_position)):
                    if new_position[i] <= 0:
                        new_position[i] += 5
                new_fitness = get_loss(input_size, output_size, new_position)
                if new_fitness < fish.fitness:
                    fish.position = new_position
                    fish.fitness = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fish = fish
                        self.best_fitness = new_fitness
                        self.best_position = new_position

    # Define the follow behavior
    def follow(self, input_size, output_size, fish):
        n_local = 0
        best_fitness = np.inf
        best_position = None
        for other in self.fishes:
            if np.linalg.norm(fish.position - other.position) < self.visual:
                n_local += 1
                if other.fitness < best_fitness:
                    best_fitness = other.fitness
                    best_position = other.position
                if best_position is not None and best_fitness < fish.fitness:
                    new_position = fish.position + (best_position - fish.position) * np.random.randint(0, 1) * self.step / np.linalg.norm(best_position - fish.position)
                    new_position = np.round(new_position).astype(int)
                    for i in range(len(new_position)):
                        if new_position[i] <= 0:
                            new_position[i] += 5
                    new_fitness = get_loss(input_size, output_size, new_position)
                    if new_fitness < fish.fitness:
                        fish.position = new_position
                        fish.fitness = new_fitness
                        if new_fitness < self.best_fitness:
                            self.best_fish = fish
                            self.best_fitness = new_fitness
                            self.best_position = new_position

    # Run the optimization process
    def optimize(self):
        for _ in range(self.n_iter):
            for fish in self.fishes:
                self.forage(input_size, output_size, fish)
                self.swarm(input_size, output_size, fish)
                self.follow(input_size, output_size, fish)
            print(self.best_position, self.best_fitness)
            self.fitness_curve.append(self.best_fitness)

        plt.plot(self.fitness_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness Curve of Fish Swarm Optimization')
        plt.show()

# Instantiate and run the fish swarm optimization
swarm = FishSwarm(n_fish=n_fish, n_iter=n_iter, visual=visual, step=step, delta=delta)
swarm.optimize()
