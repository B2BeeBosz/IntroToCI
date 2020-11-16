import numpy as np
from load_text import *
from GAMLP import *

def predict(parameters, X, D):
    # initialize Y
    Y = np.zeros((2, X.shape[1]))

    # Feed Forward
    AL = feed_forward(X, parameters)[0]
    prediction = AL
    
    # Find Max
    j = np.argmax(prediction, axis=0)
    for i in range(Y.shape[1]):
        if(j[i] == 0):
            Y[0][i] = 0.
            Y[1][i] = 1.
        else:
            Y[0][i] = 1.
            Y[1][i] = 0.

    return Y

def train_GA_MLP(file):
    # load data
    X_data, D_data = load_text(file_name)
    
    layers_dims = layer_sizes(X_data, D_data)

    # start process
    for i in range(10):
        X_train, D_train, X_test, D_test = load_data(X_data, D_data, 10, i)             # 10% Cross Validation
        X_train_norm, max_X_train, min_X_train = norm(X_train)
        parameters = GA_MLPmodel(X_train, D_train, layers_dims, 100, 100, True)    # Train
        
        X_test_norm, max_X_test, min_X_test = norm(X_test)
        predictions = predict(parameters, X_test, D_test)                       # Predict
        # predictions_denorm = denorm(D_test, max_X_test, min_X_test)
        
        error = []
        for j in range(predictions.shape[1]):
            error.append(np.sqrt((D_test[0][j] - predictions[0][j]) ** 2))
        mse = sum(error) / len(error)

        print("============== Round " + str(i+1) + " ==============")
        print("Predictions:\n" + str(predictions))
        print("Desired Output:\n" + str(D_test))
        print("Mean Square Error Average: " + str(mse))
        print("====================================================\n")


file_name = './HW3/wdbc.data'
train_GA_MLP(file_name)