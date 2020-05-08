#!/usr/bin/env python
# coding: utf-8

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris


def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    # Initialize weights list and append weights based on the size of the hidden layers
    weights = []

    # Initialize list of average errors for each epoch
    err = []

    # Initialize list for each point with the labels
    points_list = []

    # Create a list of ones for each layer in the NN for the bias terms
    all_bias_terms = np.ones(X_train.shape[0])

    # Adding the bias terms to X_train. np.newaxis is used to convert the all_bias_terms 1D array into
    # a 2D array so that we can add exactly one bias term to each later in the NN
    X_train = np.hstack((all_bias_terms[:, np.newaxis], X_train))

    # Add weights to the first layer
    weights.append(np.full([X_train.shape[1], hidden_layer_sizes[0]], 0.1))

    # Fill in the rest of the weights for the hidden layers
    for i in range(len(hidden_layer_sizes)):
        if i != len(hidden_layer_sizes) - 1:
            weights.append(np.full([hidden_layer_sizes[i] + 1, hidden_layer_sizes[i + 1]], 0.1))
        else:
            # Last hidden layer
            weights.append(np.full([hidden_layer_sizes[i] + 1, 1], 0.1))

    # Get the list with all of the X_train points with the labels
    points_list = np.column_stack((X_train, y_train))

    # Loop through the number of epochs, forward and backwards propagate, perform weight updates,
    # and keep track of the error
    for epoch in np.range(0, epochs):
        # Initialize the list of errors for the current epoch
        err_to_add = []

        # Shuffle the data points at every epoch loop
        np.random.shuffle(points_list)

        # Only take the y_train values from the shuffled data set
        y_train_shuffled = points_list[:, -1]

        # Only take the X_train values from the shuffled data set
        X_train_shuffled = points_list[:, :-1]

        # Loop through the shuffled X_train values and build the neural network
        for l in range(X_train_shuffled.shape[0]):

            # Get the forward propagation
            # We have to convert X_train_shuffled into a 2D array so we can perform multiplication of matrices
            # in the forward propagation function.
            X, S = forwardPropagation(X_train_shuffled[l, :, np.newaxis], weights)

            # Get the backwards propagation
            g = backPropagation(X, y_train_shuffled[l], S, weights)

            # Perform weight updates
            weights = updateWeights(weights, g, alpha)

            # Accumulate the error for the current epoch
            err_to_add.append(errorPerSample(X, y_train_shuffled[l]))

        # Accumulate total error from each data point based on the current epoch
        err.append(np.mean(err_to_add))

    return err, weights


def errorPerSample(X, y_n):
    # Last Item in the List X
    x_L = X[-1]
    eN = 0
    eN = errorf(x_L, y_n)
    return eN


def forwardPropagation(x, weights):
    # Initialize list for X and S
    X = []
    S = []

    # Initialize lists used to store the current layer's values
    current_s = []
    current_x = []

    # Initialize lists used to store the current layer's values with the bias
    current_x_bias = []

    # To avoid having to use "a.all()" for the activation and output functions, vectorize the functions instead
    # to account for passing a vector into the two functions and applying the function.
    outputf_list = np.vectorize(outputf)
    activation_list = np.vectorize(activation)

    # For the bias node, add the first element of x which will be 1
    X.append(x)

    for i in range(len(weights) - 1):
        # First layer
        if i == 0:
            # For the first layer, do not add the bias term on x
            current_s = np.dot(np.transpose(weights[i]), x)
        else:
            # For the rest of the layers, use the x with the bias term
            current_s = np.dot(np.transpose(weights[i]), current_x_bias)

        S.append(current_s)
        # Calculate the output x using the activation function
        current_x = activation_list(current_s)
        # Add the bias term to column number zero for each vector in X
        current_x_bias = np.insert(current_x, 0, 1, axis=0)
        X.append(current_x_bias)

    # Final layer
    current_s = np.dot(np.transpose(weights[-1]), current_x_bias)
    S.append(current_s)

    # Use the outputf function since it is the final layer
    current_x = outputf_list(current_s)
    X.append(current_x)

    return X, S


def backPropagation(X, y_n, s, weights):
    # Initialize list g for the backward message
    g = []

    # Initialize the list for the gradients of the error for each layer
    err_gradient = []

    # The second last layer value
    backLayer = 2

    # To avoid having to use "a.all()" for the activation and output functions, vectorize the functions instead
    # to account for passing a vector into the function and applying the derivative of the activation.
    derivativeActivation_list = np.vectorize(derivativeActivation)

    # Calculate delta for the last layer
    delta_l = derivativeError(X[-1], y_n) * derivativeOutput(s[-1])  # Backwards message of the last layer

    # Calculate the gradient of the error
    err_gradient = X[-backLayer] * delta_l

    # Insert the gradient of error into list g
    g.insert(0, err_gradient)

    # Backward messages for the rest of the layers
    for l in range(len(weights) - 1, 0, -1):
        # Get the derivative for each output X value in the current layer
        derivativeActivation_current = derivativeActivation_list(X[-backLayer])

        # Calculate delta for the current layer
        delta_l = derivativeActivation_current[1:] * np.dot(weights[l][1:], delta_l)

        # Calculate the gradient of the error
        err_gradient = np.dot(X[-backLayer - 1], np.transpose(delta_l))

        # Keep going back one layer for the output X
        backLayer += 1

        # Insert the gradient of error into list g
        g.insert(0, err_gradient)

    return g


def updateWeights(weights, g, alpha):
    # Initialize list for the updated weights
    nW = []

    # Loop through each layer in the NN that has weights and update the weights
    for i in range(len(weights)):
        nW.append(weights[i] - g[i] * alpha)
    return nW

def activation(s):
    # Implementation of the ReLU function. x = theta(s) = ReLU(s) = max(0, s)
    x = 0
    if s > 0:
        x = s
    return x

def derivativeActivation(s):
    # Derivative of the ReLU function
    # (i.e if ReLU(s) = 0, the derivative is zero, and if ReLU(s) = s, the derivative is one)
    x_derivative = 0
    if activation(s) > 0:
        x_derivative = 1

    return x_derivative


def outputf(s):
    x_L = 0
    # Logistic Regression Function
    x_L = 1/(1 + np.exp(-s))

    return x_L


def derivativeOutput(s):
    x_L = 0
    # Derivative of the Logistic Regression Function
    x_L = np.exp(-s)/((1 + np.exp(-s))*(1 + np.exp(-s)))
    return x_L


def errorf(x_L, y):
    # Implementation of the log loss error function
    # Initializing the indicator function (i.e if the indicator condition is true, it outputs one, else zero)
    ind_1 = 0
    ind__2 = 0
    e_n = 0

    # if y = 1, then the first indicator function outputs one and the second indicator function outputs zero
    if y == 1:
        ind_1 = 1
        ind__2 = 0

    # if y = -1, then the first indicator function outputs zero and the second indicator function outputs one
    elif y == -1:
        ind_1 = 0
        ind__2 = 1

    # Log loss error function
    e_n = -ind_1*np.log(x_L) - ind__2*np.log(1 - x_L)
    return e_n



def derivativeError(x_L, y):
    # Implementation of the log loss error function
    # Initializing the indicator function (i.e if the indicator condition is true, it outputs one, else zero
    ind_1 = 0
    ind_2 = 0
    e_n_derivative = 0

    # if y = 1, then the first indicator function outputs one and the second indicator function outputs zero
    if y == 1:
        ind_1 = 1
        ind__2 = 0

    # if y = -1, then the first indicator function outputs zero and the second indicator function outputs one
    elif y == -1:
        ind_1 = 0
        ind__2 = 1

    # Derivative of the log loss error function
    e_n_derivative = (-ind_1/x_L) - (-1*ind__2/(1 - x_L))

    return e_n_derivative


def pred(x_n, weights):

    # Initialize the variable for the output of the prediction, c
    c = 0

    # Perform forward propagation
    X, S = forwardPropagation(x_n, weights)

    # If the output of the last layer in the NN is greater than or equal to 0.5, return 1, otherwise return -1
    if X[-1] >= 0.5:
        c = 1
    else:
        c = -1
    return c


def confMatrix(X_train, y_train, w):
    # Initialize the matrix values
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0

    # Calculate how many points were correctly classified to be -1
    for i in range(len(X_train)):
        if (pred(np.hstack((np.ones(1, dtype=int), X_train[i])), w) == -1) and (y_train[i] == -1):
            class_1 = class_1 + 1

    # Calculate how many points were predicted to be classified as -1 but are actually +1
    for j in range(len(y_train)):
        if (pred(np.hstack((np.ones(1, dtype=int), X_train[j])), w) == -1) and (y_train[j] == 1):
            class_2 = class_2 + 1

    # Calculate how many points were predicted to be classified as +1 but are actually -1
    for k in range(len(y_train)):
        if (pred(np.hstack((np.ones(1, dtype=int), X_train[k])), w) == 1) and (y_train[k] == -1):
            class_3 = class_3 + 1

    # Calculate how many points were correctly classified to be +1
    for l in range(len(X_train)):
        if (pred(np.hstack((np.ones(1, dtype=int), X_train[l])), w) == 1) and (y_train[l] == 1):
            class_4 = class_4 + 1

    # Append the values to the 2x2 confusion matrix
    con = np.array([[class_1, class_2],
                    [class_3, class_4]])

    return con


def plotErr(e, epochs):
    # Create the list of epochs
    epoch_list = []

    # Make a vector for the X values of the plot from zero to epoch
    for i in range(epochs):
        epoch_list.append(i)

    # Plot the error with the X-axis being epochs and the Y-axis being the error from fit_NeuralNetwork
    plt.plot(epoch_list, e)

    # Create the X labe;
    plt.xlabel('Epochs')

    # Create the Y lavel
    plt.ylabel('Error')

    # Show the plot
    plt.show()


def test_SciKit(X_train, X_test, Y_train, Y_test):
    # Initialize the Neural Network with solver SGD, alpha = 10^-5, hidden_layer_sizes = (300, 100), and
    # random_state = 1
    cM = 0
    mlp_classifier = MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(300, 100), random_state=1)

    # Fit the training data to the neural network that was initialized
    mlp_classifier.fit(X_train, Y_train)

    # Predict the y values from the X training set
    pred_y = mlp_classifier.predict(X_test)

    # Create the confusion matrix from the predicted y values and the Y test set
    cM = confusion_matrix(Y_test, pred_y)

    return cM


def test():
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)

    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1

    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)

    plotErr(err, 100)

    cM = confMatrix(X_test, y_test, w)

    sciKit = test_SciKit(X_train, X_test, y_train, y_test)

    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)

test()

# Follow up questions:

# Our interpretation of the confusion matrix compared to the scikit-learn confusion matrix has some discrepancies
# due to the fact that we might be classifying points in a different way than sci-kit. Sci-kit might be using more
# accurate techniques by using thresholds to classify points. Furthermore, sci-kit could be using different types of
# activation functions than we are which causes the discrepancies, however most of the time the two confusion matrices
# match with minimal differences.

# As seen in the MATLAB plot, the error seems to be falling at an exponential rate as the epochs increase.
# This is due to the fact that our model is learning more efficiently at each iteration as the weights are updated,
# and due to the fact that we are randomizing the training points at each iteration, the curve is not very smooth
# towards the end.

# Changing the number of layers and nodes will affect the error. For example, if there is an increase in number of
# layers and nodes  the error carried forward from a layer before will also become larger after going through more
# nodes, assuming each node or doesnt add more errors to it , as each node may add their own error and the error grows,
# this makes the model more complex, and it would take a longer time to train the data points, with a limited test set.
# Therefore, this can cause over-fitting since the model will not generalize well.

# The choice of the activation function will make a difference, because when we use the sigmoid function, the output
# is a value between zero and one, which allows for higher accuracy when calculating backwards and forward messages
# in the neural network. If we compare this to the reLU function or other basic functions, it will be hard to train the
# nerual network since they would not generalize as well as the sigmoid function.