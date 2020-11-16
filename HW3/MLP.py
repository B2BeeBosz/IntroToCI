import numpy as np
import random

# Activation Functions
def activation(v, func):
    cache = v
    if (func == "linear"):
        A = v
    if (func == "sigmoid"):
        A = 1 / (1 + np.exp(-v))
    if (func == "relu"):
        A = np.maximum(0, v)
    
    return A, cache

# Derivative of Activation Functions
def diff_activation(dA, cache, func):
    V = cache
    if (func == "linear"):
        dV = 1
    if (func == "sigmoid"):
        s = 1 / (1 + np.exp(-V))
        dV = dA * s * (1-s)
    if (func == "relu"):
        dV = np.array(dA, copy=True)
        dV[V <= 0] = 0
    
    return dV

# Normalization
def norm(X):                             # X = (8, 282) D = (1, 282)
    Z = np.zeros((X.shape[0], X.shape[1]))  # inputs
    # Y = np.zeros((D.shape[0], D.shape[1]))  # desired output
    
    max_X = np.zeros((1, X.shape[1]))
    min_X = np.multiply(9999, np.ones((1, X.shape[1])))

    for j in range(X.shape[1]):    
        for i in range(X.shape[0]):
            if (X[i][j] > max_X[0][j]):
                max_X[0][j] = X[i][j]
            if (X[i][j] < min_X[0][j]):
                min_X[0][j] = X[i][j]
    for j in range(X.shape[1]):   
        for i in range(X.shape[0]):
            Z[i][j] = (X[i][j] - min_X[0][j]) / (max_X[0][j] - min_X[0][j])
    
    # for j in range(D.shape[1]):
    #     for i in range(D.shape[0]):
    #         Y[i][j] = (D[i][j] - min_X[0][j]) / (max_X[0][j] - min_X[0][j])
    
    return Z, max_X, min_X

def denorm(D, max_X, min_X):
    Y = np.zeros((D.shape[0], D.shape[1]))

    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            Y[i][j] = D[i][j] * (max_X[0][j] - min_X[0][j]) + min_X[0][j]
    
    return Y

# define layer sizes
def layer_sizes(X, Y):
    nx = X.shape[0]   # input layer nodes = 30
    nh = [4, 3]       # hidden layer nodes = 8, 4
    ny = Y.shape[0]   # output layer nodes = 2
    
    layers_dims = []
    layers_dims.append(nx)
    for l in range(len(nh)):
        layers_dims.append(nh[l])
    layers_dims.append(ny)    
    
    return layers_dims

# initialize population contains weight and bias
def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    # 3 populations
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

# linear forward
def linear_forward(A, W, b):
    V = np.dot(W, A) + b
    
    assert(V.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return V, cache

def linear_activation_forward(A_prev, W, b, func):
    V, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = activation(V, func)
    
    assert(A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def feed_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
                    A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "sigmoid")
        caches.append(cache)

    AL, cache = linear_activation_forward(
                    A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    assert (AL.shape == (2, X.shape[1]))
    return AL, caches

# compute cost
def calculate_cost(AL, Y): 
    m = Y.shape[1]
    
    # error from desired output
    cost = -1/m * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# back propagation to find gradient
def linear_backward(dV, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dV, A_prev.T)
    db = 1/m * np.sum(dV, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dV)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, func):
    linear_cache, activation_cache = cache
    
    dV = diff_activation(dA, activation_cache, func)
    dA_prev, dW, db = linear_backward(dV, linear_cache)

    return dA_prev, dW, db

def back_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))
    current_cache = caches[L-1]
    grads["dA"+str(L)] = dAL
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    for l in range(L-1, -1, -1):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)], current_cache, "sigmoid")
        grads["dA"+str(l)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp

    return grads

# update parameters
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - (learning_rate * grads["dW"+str(l+1)])
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - (learning_rate * grads["db"+str(l+1)])
        
    return parameters

def random_sample(X, Y):
    col = X.shape[1]
    idx = random.sample(range(0, col), col)
    X_random = np.zeros((X.shape[0], col))
    Y_random = np.zeros((Y.shape[0], col))

    for i in range(X.shape[0]):
        for j in range(col):
            X_random[i][j] = X[i][idx[j]]
    for i in range(Y.shape[0]):
        for j in range(col):
            Y_random[i][j] = Y[i][idx[j]]

    return idx, X_random, Y_random

# MLP model
def model(X, Y, layers_dims, learning_rate, max_epoch, print_cost):
    np.random.seed(1)
    epsilon = 1e-6
    
    # initialize parameters
    parameters = initialize_params(layers_dims)

    # Train
    for epoch in range(1, max_epoch+1):
        # Uniquely random train examples
        idx_x, X, Y = random_sample(X, Y)
        
        costs = []
        for idx in idx_x:
            # Train each example
            X_sample = np.array([[X[i][idx]] for i in range(X.shape[0])]) 
            Y_sample = np.array([[Y[i][idx]] for i in range(Y.shape[0])])

            # Feed Forward
            AL, caches = feed_forward(X_sample, parameters)
            # Calculate cost 
            cost = calculate_cost(AL, Y_sample)
            # Back Propagation
            grads = back_propagation(AL, Y_sample, caches)
            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

        # print cost
        if(epoch % 200 == 0 and print_cost == True):
            print("Cost after " + str(epoch) + " epoch: " + str(cost))
            costs.append(cost)
                
            if (cost < epsilon):
                break

    return parameters