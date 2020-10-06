import numpy as np
import random

np.random.seed(0)

def load_text(file):
    f = open(file, 'r')
    contents = f.readlines()
    linerows = len(contents)
    rows = int(linerows / 3)
    X_data = np.zeros((2, rows))
    D_data = np.zeros((2, rows))
    
    group = 0
    count = 0
    for line in range(linerows):
        if(group == 0):
            group = group + 1
            continue
        elif(group == 1):
            X_data[0][count] = float(contents[line].split()[0])
            X_data[1][count] = float(contents[line].split()[1])
            group = group + 1
            continue
        elif(group == 2):
            D_data[0][count] = float(contents[line].split()[0])
            D_data[1][count] = float(contents[line].split()[1])
            group = 0
            count = count + 1
            continue
    
    return X_data, D_data

def load_data(X_data, D_data, cross_percent, rounds):
    train_percent = 100 - cross_percent
    test_percent = cross_percent
    rows = X_data.shape[1]
    train = int(np.round(train_percent * rows / 100))
    test = int(np.round(test_percent * rows / 100))

    # Slice all data into piles
    slices = int(np.round(100/cross_percent))
    piles_X = np.zeros((slices, X_data.shape[0], test))
    piles_D = np.zeros((slices, D_data.shape[0], test))

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

# define layer sizes
def layer_sizes(X, Y):
    nx = X.shape[0]   # input layer nodes = 2
    nh = 4            # hidden layer nodes = 4
    ny = Y.shape[0]   # output layer nodes = 2
    
    return nx, nh, ny

# define activation function each layer
def layer_activ_func(name1, name2):
    func = (name1, name2)
    return func

# initialize weight and bias
def initialize_params(nx, nh, ny):
    W1 = np.random.randn(nh, nx)       # (4 x 2)
    W2 = np.random.randn(ny, nh)       # (2 x 4)
    b1 = np.zeros((nh, 1))
    b2 = np.zeros((ny, 1))

    print(W1)
    print(W2)
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

    V1 = np.dot(W1, X) + b1         # calculate in hidden layer             (4 x 2) (2 x 1) + (4 x 1) = (4 x 1)
    A1 = activation(V1, func[0])    # hidden layer activation function
    V2 = np.dot(W2, A1) + b2        # calculate in output layer             (2 x 4) (4 x 1) + (2 x 1) = (2 x 1)
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
    E = Y - A2                                          # (2 x 1)
    sqe = 1/2 * np.sum(np.multiply(E, E))

    return sqe, E

# back propagation to find local gradient
def backward_propagation(params, result, E, X, Y, func):
    W2 = params["W2"]       # (2 x 4)
    A1 = result["A1"]       # (4 x 1)
    A2 = result["A2"]       # (2 x 1)
    
    # find local gradient
    dZ2 = np.multiply(E, diff_activation(A2, func[1]))                      # (2 x 1) * (2 x 1)
    LG2 = np.multiply(dZ2, X)           # -dSQE / dw2 = local gradient of output layer  (2 x 1) * (2 x 1) = (2 x 1)
    LGB2 = dZ2                          # -dSQE / db2 = (2 x 1)
    
    dZ1 = np.multiply(np.dot(LG2.T, W2).T, diff_activation(A1, func[0]))    # (4 x 1) * (4 x 1)
    dZb1 = np.multiply(np.dot(LGB2.T, W2).T, diff_activation(A1, func[0]))  # (4 x 1) * (4 x 1)
    LG1 = dZ1                           # -dSQE / dw1 = local gradient of hidden layer  (4 x 1)
    LGB1 = dZb1                         # -dSQE / db1 = (4 x 1)
    
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

    # update weight
    delta_W2 = np.multiply(momentum_rate, delta_W2_prev) + np.multiply(learning_rate, np.dot(LG2, A1.T)) # (2 x 4) + (2 x 1) (1 x 4)
    W2 = W2 + delta_W2
    delta_W2_prev = delta_W2

    delta_W1 = np.multiply(momentum_rate, delta_W1_prev) + np.multiply(learning_rate, np.dot(LG1, X.T))  # (4 x 2) + (4 x 1) (1 x 2)
    W1 = W1 + delta_W1
    delta_W1_prev = delta_W1

    # update bias
    delta_b2 = np.multiply(momentum_rate, delta_b2_prev) + np.multiply(learning_rate, LGB2) # (2 x 1) + (2 x 1)
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
    
# predict
def predict(params, X, D, func):
    # initialize Y
    Y = np.zeros((2, X.shape[1]))

    # Feed Forward
    result = forward_propagation(params, X, func)
    prediction = result["A2"]
    
    # Find Max
    j = np.argmax(prediction, axis=0)
    for i in range(Y.shape[1]):
        if(j[i] == 0):
            Y[0][i] = 1.
            Y[1][i] = 0.
        else:
            Y[0][i] = 0.
            Y[1][i] = 1.

    return Y

def confusionMatrix(Y, D, print_confusion, conf_count):
    tp = conf_count["tp"] 
    tn = conf_count["tn"]
    fp = conf_count["fp"]
    fn = conf_count["fn"]

    for j in range(Y.shape[1]):
        # Correct Prediction
        if ((D[0][j] == 0.) and (Y[0][j] == 0.)):
            tn = tn + 1                    
        elif ((D[0][j] == 1.) and (Y[0][j] == 1.)):
            tp = tp + 1
        # Wrong Prediction
        elif ((D[0][j] == 0.) and (Y[0][j] == 1.)):
            fp = fp + 1
        elif ((D[0][j] == 1.) and (Y[0][j] == 0.)):
            fn = fn + 1

    # Return newly counted counters
    conf_count = {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

    # print confusion matrix
    if(print_confusion == True):
        # compute accuracy
        accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
        
        print("Prediction\t|\t[1 0]\t|\t[0 1]\t|")
        print("Actual")
        print("[1 0]\t\t|\t" + str(tp) + "\t|\t" + str(fn) + "\t|\t" + str(tp + fn))
        print("[0 1]\t\t|\t" + str(fp) + "\t|\t" + str(tn) + "\t|\t" + str(fp + tn))
        print("\t\t|\t" + str(tp + fp) + "\t|\t" + str(fn + tn) + "\t|\t" + str(tp+fp+tn+fn) + "\n")
        print("===========================================================\n")
        return accuracy
    else:
        return conf_count

def MLP(file_name, learning_rate, momentum_rate):
    # set activation function
    func = layer_activ_func("sigmoid", "sigmoid")
    
    # load data
    X_data, D_data = load_text(file_name)

    # initialize confusion matrix counter
    conf_count = {
        "tp": 0, "tn": 0, "fp": 0, "fn": 0
    }
    
    # start process
    for i in range(10):
        X_train, D_train, X_test, D_test = load_data(X_data, D_data, 10, i)         # 10% Cross Validation
        params = model(X_train, D_train, 3000, learning_rate, momentum_rate, True)  # Train
        predictions = predict(params, X_test, D_test, func)                         # Predict
        
        error = []
        for j in range(predictions.shape[1]):
            error.append(np.sqrt((D_test[0][j] - predictions[0][j]) ** 2))
        mse = sum(error) / len(error)

        print("============== Round " + str(i+1) + " ==============")
        print("Predictions:\n" + str(predictions))
        print("Desired Output:\n" + str(D_test))
        print("Mean Square Error Average: " + str(mse))
        print("====================================================\n")

        if (i == 9):
            accuracy = confusionMatrix(predictions, D_test, True, conf_count)
        else:
            conf_count = confusionMatrix(predictions, D_test, False, conf_count)
    
    # Conclude the experiment
    print("Total Accuracy: " + str(accuracy) + "\n")
    print("===========================================================")

file_name = "cross.pat"
learning_rate = 0.06
momentum_rate = 0.09
MLP(file_name, learning_rate, momentum_rate)