## This is based on the simple perceptron code from class but modified to work with images

import numpy as np

## Load Dataset (images in the LTdata folder were converted to npy locally so it would work on the web)
def load_dataset():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return X, y

## Perceptron Training
def fit( X_train, y_train ):
    learning_rate = 0.01
    n_samples, n_features = X_train.shape
    
    weights = np.zeros(n_features)
    bias = 0
    
    y_ = np.array([1 if i > 0 else 0 for i in y_train])    
    
    for _ in range(1000):
        
        for idx, x_i in enumerate(X_train):
            
            linear_output  = np.dot(x_i, weights) + bias
            y_pred = activation_function(linear_output)
            
            update = learning_rate * (y_[idx] - y_pred)
            
            weights = weights + update * x_i
            bias = bias + update
    
    return weights, bias
                           
########################################

def activation_function(x):
    return np.where(x>=0, 1, 0)
     
########################################
    
def predict(X_test, weights, bias):
    linear_output = np.dot(X_test, weights) + bias
    y_pred = activation_function(linear_output)
    return y_pred

#######################################

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

#######################################

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

#######################################

X, y = load_dataset()

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#######################################

weights, bias = fit(X_train, y_train)
print("weights shape:", weights.shape)
print("bias shape:", bias.shape)
np.save("weights.npy", weights)
np.save("bias.npy", np.array([bias]))
y_pred = predict(X_test, weights, bias)

#######################################

## compare y_pred to y_test

print("Accuracy: " + str(accuracy(y_test, y_pred)))

print("*********************************")
print("actual:     " + str(y_test[:20]))
print("prediction: " + str(y_pred[:20]))