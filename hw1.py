import numpy as np
import random

np.random.seed(0)

def load_text(file):
    X_data = np.loadtxt(file, dtype=float, delimiter='\t', skiprows=2, usecols={0,1,2,3,4,5,6,7}, unpack=True)
    D_data = np.loadtxt(file, dtype=float, delimiter='\t', skiprows=2, usecols={8}, unpack=True)
    
    D_data = np.reshape(D_data, (1, D_data.shape[0]))
    return X_data, D_data

# load data 10% cross validation -> 90% train examples
def load_data(X_data, D_data, cross_percent, rounds):
    train_percent = 100 - cross_percent
    test_percent = cross_percent
    rows = X_data.shape[1]
    train = int(np.round(train_percent * rows / 100) - 4)
    test = int(np.round(test_percent * rows / 100))

    # Slice all data into piles
    slices = int(np.round(100/cross_percent))
    piles_X = np.zeros((slices, X_data.shape[0], test))
    piles_D = np.zeros((slices, 1, test))

    for i in range(piles_X.shape[0]):
        for j in range(piles_X.shape[1]):
            for k in range(piles_X.shape[2]):
                piles_X[i][j][k] = X_data[j][k + i * piles_X.shape[2]]

    for i in range(piles_D.shape[0]):
        for j in range(piles_D.shape[1]):
            for k in range(piles_D.shape[2]):
                piles_D[i][j][k] = D_data[j][k + i*piles_D.shape[2]]

    # initialize vector
    train_data_set_X = np.zeros((X_data.shape[0], train))
    test_data_set_X = np.zeros((X_data.shape[0], test))
    train_data_set_D = np.zeros((D_data.shape[0], train))
    test_data_set_D = np.zeros((D_data.shape[0], test))
    
    # Set Train and Test Sample
    for j in range(piles_X.shape[1]):
        for k in range(piles_X.shape[2]):
            test_data_set_X[j][k] = piles_X[rounds][j][k]
    for j in range(piles_D.shape[1]):
        for k in range(piles_D.shape[2]):
            test_data_set_D[j][k] = piles_D[rounds][j][k]
    
    step = 0
    for i in range(piles_X.shape[0]):
        if (i == rounds):
            continue
        for j in range(piles_X.shape[1]):
            for k in range(piles_X.shape[2]):
                train_data_set_X[j][k + step*piles_X.shape[2]] = piles_X[i][j][k]
        step += 1
    
    step = 0
    for i in range(piles_D.shape[0]):
        if (i == rounds):
            continue
        for j in range(piles_D.shape[1]):
            for k in range(piles_D.shape[2]):
                train_data_set_D[j][k + step*piles_X.shape[2]] = piles_D[i][j][k]
        step += 1

    X_train = train_data_set_X
    D_train = train_data_set_D
    X_test = test_data_set_X
    D_test = test_data_set_D    

    return X_train, D_train, X_test, D_test

# Activation Functions
def activation(v, func):
    if (func == "linear"):
        return v
    if (func == "sigmoid"):
        return 1 / (1 + np.exp(-v))
    if (func == "tanh"):
        return np.tanh(v)

# Derivative of Activation Functions
def diff_activation(v, func):
    if (func == "linear"):
        return 1
    if (func == "sigmoid"):
        return np.multiply(v, 1-v)
    if (func == "tanh"):
        return 2 * diff_activation(v, "sigmoid")

# Normalization
def norm(X, D):                             # X = (8, 282) D = (1, 282)
    Z = np.zeros((X.shape[0], X.shape[1]))  # inputs
    Y = np.zeros((D.shape[0], D.shape[1]))  # desired output
    
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
    
    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            Y[i][j] = (D[i][j] - min_X[0][j]) / (max_X[0][j] - min_X[0][j])
    
    return Z, Y, max_X, min_X

def denorm(D, max_X, min_X):
    Y = np.zeros((D.shape[0], D.shape[1]))

    for j in range(D.shape[1]):
        for i in range(D.shape[0]):
            Y[i][j] = D[i][j] * (max_X[0][j] - min_X[0][j]) + min_X[0][j]
    
    return Y

# define layer sizes
def layer_sizes(X, Y):
    nx = X.shape[0]   # input layer nodes = 8
    nh = 3            # hidden layer nodes = 4
    ny = Y.shape[0]   # output layer nodes = 1
    
    return nx, nh, ny

# define activation function each layer
def layer_activ_func(name1, name2):
    func = (name1, name2)
    return func

# initialize weight and bias
def initialize_params(nx, nh, ny):
    W1 = np.random.randn(nh, nx)       # (4 x 8)
    W2 = np.random.randn(ny, nh)       # (1 x 4)
    b1 = np.zeros((nh, 1))
    b2 = np.zeros((ny, 1))

    params = {
        "W1": W1, "W2": W2, "b1": b1, "b2": b2    
    }
    
    return params

# calculate on each layer
def forward_propagation(params, X, func):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]

    V1 = np.dot(W1, X) + b1         # calculate in hidden layer             (4 x 8) (8 x 1) + (4 x 1) = (4 x 1)
    A1 = activation(V1, func[0])    # hidden layer activation function
    V2 = np.dot(W2, A1) + b2        # calculate in output layer             (1 x 4) (4 x 1) + (1 x 1)= (1 x 1)
    A2 = activation(V2, func[1])    # output layer activation function
    
    result = {
        "V1": V1, "A1": A1, "V2": V2, "A2": A2
    }

    return result

# compute cost
def calculate_cost(result, params, Y):
    A2 = result["A2"]
    m = Y.shape[1]
    
    # error from desired output
    E = Y - A2                                          # (1 x 1)
    sqe = 1/2 * np.sum(np.multiply(E, E))

    return sqe, E

# back propagation to find local gradient
def backward_propagation(params, result, E, X, Y, func):
    W2 = params["W2"]       # (1 x 4)
    A1 = result["A1"]       # (4 x 1)
    A2 = result["A2"]       # (1 x 1)
    
    # find local gradient
    LG2 = np.dot(E, diff_activation(A2, func[1]))                          # -dSQE / dw2 = local gradient of output layer  (1 x 1) (1 x 1) = (1 x 1)
    LG1 = np.multiply(diff_activation(A1, func[0]), np.dot(W2.T, LG2))     # -dSQE / dw1 = local gradient of hidden layer  (4 x 1) * (4 x 1) = (4 x 1)
    LGB2 = np.dot(E, diff_activation(A2, func[1]))                         # -dSQE / db2 = (1 x 1) (1 x 1) = (1 x 1)
    LGB1 = np.dot(diff_activation(A1, func[0]), LGB2)                      # -dSQE / db1 = (4 x 1) (1 x 1) = (4 x 1)
    # print(LGB1.shape)

    local_grads = {
        "LG1": LG1, "LG2": LG2, "LGB1": LGB1, "LGB2": LGB2
    }

    return local_grads

# update parameters
def update(X, params, result, local_grads, delta_prev, learning_rate, momentum_rate):
    # Retrieve parameters and cache
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]
    A1 = result["A1"]
    A2 = result["A2"]
    # Retrieve local gradients
    LG1 = local_grads["LG1"]
    LG2 = local_grads["LG2"]
    LGB1 = local_grads["LGB1"]
    LGB2 = local_grads["LGB2"]

    # Retrieve weight from last round
    delta_W1_prev = delta_prev["delta_W1_prev"]
    delta_W2_prev = delta_prev["delta_W2_prev"]
    delta_b1_prev = delta_prev["delta_b1_prev"]
    delta_b2_prev = delta_prev["delta_b2_prev"]

    # print("W1\n" + str(delta_W1_prev))
    # print("LG1\n" + str(LG1))
    # print("X\n" + str(X))

    # update weight
    delta_W2 = np.multiply(momentum_rate, delta_W2_prev) + np.multiply(learning_rate, np.dot(LG2, A1.T)) # (1 x 4) + (1 x 1) (1 x 4)
    W2 = W2 + delta_W2
    delta_W2_prev = delta_W2

    delta_W1 = np.multiply(momentum_rate, delta_W1_prev) + np.multiply(learning_rate, np.dot(LG1, X.T))  # (4 x 8) + (4 x 1) (1 x 8)
    W1 = W1 + delta_W1
    delta_W1_prev = delta_W1

    # update bias
    delta_b2 = np.multiply(momentum_rate, delta_b2_prev) + np.multiply(learning_rate, LGB2) # (1 x 1) + (1 x 1)
    b2 = b2 + delta_b2
    delta_b2_prev = delta_b2

    delta_b1 = np.multiply(momentum_rate, delta_b1_prev) + np.multiply(learning_rate, LGB1) # (4 x 1) + (4 x 1)
    b1 = b1 + delta_b1
    delta_b1_prev = delta_b1

    # return newly updated weights and bias
    params = {
        "W1": W1, "W2": W2, "b1": b1, "b2": b2
    }
    delta_prev = {
        "delta_W1_prev": delta_W1_prev, "delta_W2_prev": delta_W2_prev, "delta_b1_prev": delta_b1_prev, "delta_b2_prev": delta_b2_prev
    }

    return params, delta_prev

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
def model(X, Y, max_epoch, learning_rate, momentum_rate, print_cost):
    # initialize parameters
    nx, nh, ny = layer_sizes(X, Y)
    func = layer_activ_func("sigmoid", "sigmoid")
    params = initialize_params(nx, nh, ny)
    delta_prev = {
        "delta_W1_prev": np.zeros((nh, nx)),
        "delta_W2_prev": np.zeros((ny, nh)),
        "delta_b1_prev": np.zeros((nh, 1)),
        "delta_b2_prev": np.zeros((ny, 1))
    }
    m = Y.shape[1]
    sum_square_error_prev = 1.
    
    for epoch in range(1, max_epoch+1):
        # Uniquely random train examples
        idx_x, X, Y = random_sample(X, Y)
        
        sqe = []
        for idx in idx_x:
            # Train each example
            X_sample = np.array([[X[i][idx]] for i in range(X.shape[0])]) 
            Y_sample = np.array([[Y[i][idx]] for i in range(Y.shape[0])])

            # Feed Forward
            result = forward_propagation(params, X_sample, func)
            # Calculate cost 
            sqe.append(calculate_cost(result, params, Y_sample)[0]) 
            E = calculate_cost(result, params, Y_sample)[1]
            # Back Propagation
            local_grads = backward_propagation(params, result, E, X_sample, Y_sample, func)
            # Update parameters
            params, delta_prev = update(X_sample, params, result, local_grads, delta_prev, learning_rate, momentum_rate)

        # print cost
        sum_square_error = sum(sqe) / len(sqe)
        if(epoch % 500 == 0 and print_cost == True):
            print("Cost after " + str(epoch) + " epoch: " + str(sum_square_error))
            
            if(sum_square_error > sum_square_error_prev):   # prevent overfitting
                break
            sum_square_error_prev = sum_square_error
            
    return params
    
# predict water level at nawarat river in the next 1 hour
def predict(params, X, D, func):
    # initialize Y
    Y = np.zeros((1, X.shape[1]))

    # Feed Forward
    result = forward_propagation(params, X, func)
    Y = result["A2"]

    return Y

def MLP(file_name, learning_rate, momentum_rate):
    # set activation function
    func = layer_activ_func("sigmoid", "sigmoid")
    
    # load data
    X_data, D_data = load_text(file_name)

    mse = []
    # start process
    for i in range(10):
        X_train, D_train, X_test, D_test = load_data(X_data, D_data, 10, i)                 # 10% Cross Validation
        X_train_norm, D_train_norm, max_X_train, min_X_train = norm(X_train, D_train)       # Normalize Train
        
        params = model(X_train_norm, D_train_norm, 3000, learning_rate, momentum_rate, True)  # Train
        
        X_test_norm, D_test_norm, max_X_test, min_X_test = norm(X_test, D_test)             # Normalize Test
        
        predictions = predict(params, X_test_norm, D_test_norm, func)                       # Predict
        predictions_denorm = denorm(predictions, max_X_test, min_X_test)                    # Denormalize Test
        
        error = []
        for j in range(predictions_denorm.shape[1]):
            error.append(np.sqrt((D_test[0][j] - predictions_denorm[0][j]) ** 2))
        mse.append(sum(error) / len(error))

        print("============== Round " + str(i+1) + " ==============")
        print("Next 7 hours:\n" + str(predictions_denorm))
        print("Desired Output:\n" + str(D_test))
        print("Mean Square Error Average: " + str(mse[i]))
        print("====================================================\n")
        
    mse_av = sum(mse) / len(mse)
    print("Total Average: " + str(mse_av))

file_name = "Flood_dataset.txt"
learning_rate = 0.12
momentum_rate = 0.15
MLP(file_name, learning_rate, momentum_rate)