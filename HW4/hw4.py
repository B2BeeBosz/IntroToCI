import numpy as np
from load_text import *
from PSOMLP import *

def predict(parameters, X, D, T):
    # initialize Y
    Y = np.zeros((1, X.shape[1]))

    # Feed Forward
    AL = feed_forward(X, parameters)[0]
    # Predict 5 days ahead
    Y = AL

    return Y

def train_PSO_MLP(file):
    # load data
    X_data, D_data, T_data = load_text(file_name)
    
    layers_dims = layer_sizes(X_data, D_data)

    # start process
    for i in range(10):
        X_train, D_train, X_test, D_test = load_data(X_data, D_data, 10, i)             # 10% Cross Validation
        X_train_norm, max_X_train, min_X_train = norm(X_train)
        parameters = model(X_train, D_train, layers_dims, learning_rate, 3000, True)    # Train
        
        X_test_norm, D_test_norm, max_X_test, min_X_test = norm(X_test, D_test)
        predictions = predict(parameters, X_test_norm, D_test_norm, T_data)             # Predict
        predictions_denorm = denorm(D_test, max_X_test, min_X_test)
        
        error = []
        for j in range(predictions.shape[1]):
            error.append(np.abs((D_test[0][j] - predictions[0][j])))
        mae = sum(error) / len(error)

        print("============== Round " + str(i+1) + " ==============")
        print("Predictions:\n" + str(predictions))
        print("Desired Output:\n" + str(D_test))
        print("Mean Absolute Error Average: " + str(mae))
        print("====================================================\n")


filename = './HW4/AirQualityUCI.csv'
train_PSO_MLP(file_name)