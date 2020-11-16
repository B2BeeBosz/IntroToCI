import numpy as np

def load_text(file):
    cols = [3,6,8,10,11,12,13,14]
    X_data = np.genfromtxt(file, dtype=float, delimiter=',', skip_header=1, usecols=cols, unpack=True)
    D_data = np.genfromtxt(file, dtype=float, delimiter=',', skip_header=1, usecols={5}, unpack=True)
    T_data = np.genfromtxt(file, dtype=str, delimiter=',', skip_header=1, usecols={0,1}, unpack=True)

    D_data = np.reshape(D_data, (1, D_data.shape[0]))
    T_data = np.reshape(T_data, (T_data.shape[1], 2))

    return X_data, D_data, T_data

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
    
    # print(piles_X.shape)
    # print(train_data_set_X.shape)
    # print(test_data_set_X.shape)

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
        else:   
            for j in range(piles_X.shape[1]):
                for k in range(piles_X.shape[2]):
                    train_data_set_X[j][k + step*piles_X.shape[2]] = piles_X[i][j][k]
        step += 1
    
    step = 0
    for i in range(piles_D.shape[0]):
        if (i == rounds):
            continue
        else:
            for j in range(piles_D.shape[1]):
                for k in range(piles_D.shape[2]):
                    train_data_set_D[j][k + step*piles_X.shape[2]] = piles_D[i][j][k]
        step += 1

    X_train = train_data_set_X
    D_train = train_data_set_D
    X_test = test_data_set_X
    D_test = test_data_set_D    

    return X_train, D_train, X_test, D_test


X_data, D_data, T_data = load_text("./HW4/AirQualityUCI.csv")
# X_train, D_train, X_test, D_test = load_data(X_data, D_data, 10, 0)
# print(X_data.shape)
# print(D_data.shape)
print(T_data.shape)
# print(D_data)
# print(X_train.shape)
# print(D_train.shape)
# print(X_test.shape)
# print(D_test.shape)