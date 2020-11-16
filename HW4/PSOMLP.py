import numpy as np
import random
from MLP import activation, diff_activation, norm, denorm, layer_sizes, random_sample, feed_forward

# Initialize population and population_size
def initialize_population(X, Y, layers_dims):
    POPULATION_SIZE = 0
    for l in range(len(layers_dims)):
        POPULATION_SIZE += (layers_dims[l] * layers_dims[l+1])

    population = {}
    # assign weight and bias into blocks
    population["W"] = np.random.randn((1, POPULATION_SIZE)) * 0.01
    population["b"] = np.zeros((1, POPULATION_SIZE))

    return population, POPULATION_SIZE

# initialize parameters
def initialize_parameters(X, Y, POPULATION_SIZE):
    PBest = {}
    GBest = {}
    velocity = {}
    hparameters = {}

    PBest["W"] = np.zeros((1, POPULATION_SIZE)) + 9999
    GBest["b"] = 9999
    velocity["W"] = np.random.randn((1, POPULATION_SIZE)) * 0.01
    
    hparameters["r1"] = np.random.uniform(0.0, 1.0)
    hparameters["r2"] = np.random.uniform(0.0, 1.0)
    hparameters["c1"] = np.random.randint(0, 4)
    hparameters["c2"] = np.random.randint(0, 4)
    while(hparameters["c1"] + hparameters["c2"] > 4):
        hparameters["c1"] = np.random.randint(0, 4)
        hparameters["c2"] = np.random.randint(0, 4)
    
    hparameters["rho1"] = hparameters["r1"] * hparameters["c1"]
    hparameters["rho2"] = hparameters["r2"] * hparameters["c2"]

    return hparameters, PBest, GBest, velocity

# Calculate Cost: Objective Function
def calculate_cost(AL, Y):
    cost = np.sin(AL) * np.sin(Y) * np.sqrt(AL * Y)
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

# Backprop: Global Best
def global_best(PBest, GBest, caches, cost):
    W = caches[1]
    B = caches[2]

    for w in W:
        # Personal Best
        for pbest in PBest["W"]:
            if (cost < pbest):
                pbest = cost
                PBest["W"] = w
        # Global Best
        if (cost < GBest["W"]):
            gbest = cost
            GBest["W"] = w

    for b in B:
        # Personal Best
        for pbest in PBest["b"]:
            if (cost < pbest):
                pbest = cost
                PBest["b"] = b
        # Global Best
        if (cost < GBest["b"]):
            gbest = cost
            GBest["b"] = b
    
    return PBest, GBest

# Update velocity on each particle
def update_velocity(velocity, caches, PBest, GBest, hparameters):
    W = caches[1]
    B = caches[2]
    rho1 = hparameters["rho1"]
    rho2 = hparameters["rho2"]

    for w in W:
        velocity["W"] = velocity["W"] + (rho1 * (PBest["W"] - w)) + (rho2 * (GBest["W"] - w))
    for b in B:
        velocity["b"] = velocity["b"] + (rho1 * (PBest["b"] - b)) + (rho2 * (GBest["b"] - b))
    
    return velocity

# Update weights and bias
def update_population(population, velocity):
    population["W"] = population["W"] + velocity["W"]
    population["b"] = population["W"] + velocity["b"]
    
    return population

# MLP Model using PSO
def PSOmodel(X, Y, layers_dims, learning_rate, max_epoch, print_cost):
    np.random.seed(1)
    epsilon = 1e-6

    # initialize population
    population, POPULATION_SIZE = initialize_population(X, Y, layers_dims)

    # initialize parameters
    hparameters, PBest, GBest, velocity = initialize_parameters(X, Y, POPULATION_SIZE)

    for epoch in range(1, max_epoch+1):
        # Uniquely random train examples
        idx_x, X, Y = random_sample(X, Y)
        
        costs = []
        for idx in idx_x:
            # Train each example
            X_sample = np.array([[X[i][idx]] for i in range(X.shape[0])]) 
            Y_sample = np.array([[Y[i][idx]] for i in range(Y.shape[0])])

            # Reshape population to FFN
            population_reshaped = split_reshape_population(population_selected, layers_dims)
            # Feed Forward
            AL, caches = feed_forward(X_sample, population_reshaped)
            # Calculate cost
            cost = calculate_cost(AL, Y_sample)
            # Global Best
            PBest, GBest = global_best(PBest, GBest, caches, cost)
            # Update Velocity
            velocity = update_velocity(velocity, caches, PBest, GBest, hparameters)
            # Update Population
            population = update_population(population, velocity)

        # print cost
        if(epoch % 200 == 0 and print_cost == True):
            print("Cost after " + str(epoch) + " epoch: " + str(cost))
            costs.append(cost)
                
            if (cost < epsilon):
                break
    
    return population