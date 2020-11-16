import numpy as np
import random
from MLP import activation, diff_activation, norm, denorm, layer_sizes, random_sample, feed_forward, calculate_cost

def initialize_population(X, Y, layer_dims, max_population, max_generation):
    # initialize population size
    POPULATION_SIZE = 0
    for l in range(len(layer_dims)):
        POPULATION_SIZE += (layer_dims[l] * layer_dims[l+1])

    chromosomes = {}
    # assign genes (weights) into chromosomes
    # chromosomes[generation_No] = (population_No, POPULATION_SIZE)
    for l in range(1, max_generation+1):
        chromosomes["W"+str(l)] = np.random.randn((max_population, POPULATION_SIZE)) * 0.01

    return chromosomes, POPULATION_SIZE

# Calculate fitness function
def calculate_fitness(cost):
    accuracy = 0
    
    sqe = 0
    for i in range(X.shape[1]):
        

    # while(Y >= 0 and Y <= np.pi):
    #     if (Y < 0):
    #         Y = Y + np.pi
    #     elif (Y > np.pi):
    #         Y = Y - np.pi
    
    # fitness = Y + np.abs(np.sin(256 * Y))    # 8 bits per 1 real number
    return fitness

# Selection: Linear Ranking
def linear_ranking_selection(chromosomes):
    # initial value
    N = chromosomes.shape[1]
    Max = 1.2
    Min = 2 - Max
    
    # initial probability
    P = []
    for r in chromosomes:
        prob = 1/N * (Min + (Max - Min)*((r - 1)/(N-1)))
        P.append(prob)
    
    # find expected values
    ni = []
    for p in P:
        ni.append(p * N)

    # Stochastic universal sampling
    mating_pool = []
    index = []
    ptr = np.random.uniform(0.0, 1.0)
    sum_pi = 0
    
    for idx in range(N):
        sum_pi += ni[idx]
        while (sum_pi > ptr):
            index.append(idx)
            ptr += 1
    
    for idx in index:
        mating_pool.append(chromosomes[0][idx])

    return mating_pool

# Crossover: T-point crossover
def crossover(T, mating_pool, chromosomes):    
    L = chromosomes["W1"].shape[1]
    
    # weight population 1
    pc = 0.75
    mating_pool_new = []
    for c in chromosomes["W1"]:
        q = np.random.uniform(0.0, 1.0)
        if (q < pc):
            mating_pool_new.append(c)

    m_selected = len(mating_pool_new)

    # if m_selected is even randomly pair
    if (m_selected % 2 == 0):
        couple = random.sample(mating_pool_new, 2)
    # if m_selected is odd randomly add another chromosomes from P1
    else:
        idx = np.random.randint(0, chromosomes["W1"].shape[1])
        mating_pool_new.append(chromosomes["W1"][0][idx])
        couple = random.sample(mating_pool_new, 2)
    
    # Crossover
    # initialize k = index of crossing over
    if (T >= 2):
        k = np.random.randint(1, L, size=T)
    else:
        k = np.random.randint(1, L)

    descendant = 
    
    
    
    
    
    
    return new_chromosomes

# Mutation:
def mutation():
    return pass


def GA_MLPmodel(X, Y, layers_dims, max_population, max_generation, print_fitness):
    np.random.seed(1)
    epsilon = 1e-6

    # initialize population (chromosomes)
    chromosomes, POPULATION_SIZE = initialize_population(X, Y, layer_dims, max_population, max_generation)

    # Train (1 epoch = 1 generation)
    for generation in range(1, max_generation+1):
        # Uniquely random train examples
        idx_x, X, Y = random_sample(X, Y)
        
        fitnesses = []
        for idx in idx_x:
            # Train each example
            X_sample = np.array([[X[i][idx]] for i in range(X.shape[0])]) 
            Y_sample = np.array([[Y[i][idx]] for i in range(Y.shape[0])])

            # Calculate fitness on each individual in population[p]
            for p in range(1, max_population+1):
                AL, cache = feed_forward(X_sample, p)
                cost = calculate_cost(AL, Y_sample)
                fitness = calculate_fitness(cost)
                fitnesses.append(fitness)
            
        # Select Population
        mating_pool = linear_ranking_selection(chromosomes)
        # Crossover
        new_chromosomes = crossover(mating_pool, chromosomes)
        # Mutation
        next_gen_chromosomes = mutation(new_chromosomes)

        # print cost
        if (generation % 10 == 0 and print_fitness == True):
            print("Fitness after generation " + str(generation) + ": " + str(fitness))
            fitnesses.append(fitness)

            if (fitness < epsilon):
                break

    return chromosomes