import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# Haider Shoaib


def fit_LinRegr(X_train, y_train):
    
    # create a vector of 1's for the offset value on X_train and append it 
    ones_to_add = np.ones(X_train.shape[0], dtype = int)
    ones_to_add_inverse = ones_to_add.reshape(X_train.shape[0], 1)
    x = np.hstack((ones_to_add_inverse, X_train))
    
    # Calculation for vector w. w = (X^TX)X^TY
    first_term = np.linalg.inv(np.dot(np.transpose(x), x))
    second_term = np.dot(np.transpose(x), y_train)
    w = np.dot(first_term, second_term)
    
    return w


def mse(X_train, y_train, w):
    
    #Initialize the error variable
    err = 0
    
    # Sum the mean square value for each row in X_train and divide the total
    # by the total number of training points
    for i in range(len(X_train)):
        err = err + ((pred(np.hstack((np.ones(1, dtype = int), X_train[i])), w) - 
                      y_train[i])*(pred(np.hstack((np.ones(1, dtype = int), X_train[i])), w) - y_train[i]))
    return (err/len(X_train))

def pred(X_i,w):
    
    # Take the dot product of the feature vector and weight vector
    return np.dot(X_i, w)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    
    #Initialize the linear regression model
    lrg = linear_model.LinearRegression()
    
    # Train the training data points
    lrg.fit(X_train, Y_train)
    
    # Get the predicted output values
    y_pred = lrg.predict(X_test)
    
    # Calculate the mean squared error using the original testing points and
    # the predicted output
    mse_val = mean_squared_error(Y_test, y_pred)
    return mse_val

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    #Testing Part 2a
    w=fit_LinRegr(X_train, y_train)
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

testFn_Part2()

# The output of my implementation is very close to the scikit-learn library,
# except for a few decimal places due to rounding. The reason why
# my implementation is very close is because linear regression uses a non-iterative
# technique to get the error to be close to zero and the formula to calculate
# the weight is very accurate so there are less chances of errors occurring.
