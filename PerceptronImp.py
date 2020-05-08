import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

# Haider Shoaib

def fit_perceptron(X_train, y_train):
    
    # Initialize the weight vector with zeros
    w = np.zeros(len(X_train[0]) + 1, dtype = int)
    
    # Set the old and new weghts to the zero vector and initialize the flag
    w_old = w
    w_new = w
    flag = 0
    
    # Loop for a maxiumum of 1000 iterations (will take a significant amount
    # of time to compute since it goes through 1000 loops)
    for j in range(1000):
        
            # Loop through every element of X_train
            for i in range(len(X_train)):
                dt = np.dot(np.hstack((np.ones(1, dtype = int), X_train[i])), w)
                
                # If the point is misclassified (i.e. dt = 0) then update w
                if dt == 0:
                    w_old = w
                    w_new = w + y_train[i]*np.hstack((np.ones(1, dtype = int), X_train[i]))
                    w = w_new
                    
                    # The flag ensures that the weight is not updated again after calculating the error
                    flag = 1
                    
                    # point back to X_train[0] to test the new weight
                    i = 0
                
                else:
                    flag = 0
                
                
                # If there was no update earlier, calculate the errors with the old and new
                # weight vectors, and use the one with the lower error
                if flag == 0:
                    w_new = w + y_train[i]*np.hstack((np.ones(1, dtype = int), X_train[i]))
                    
                if errorPer(X_train, y_train, w_new) <= errorPer(X_train, y_train, w_old):
                        w = w_new
                                  
                
    return w
    
def errorPer(X_train,y_train,w):
    
    # Initialize the counter and average variables
    count = 0
    avg = 0
    
    # Loop through the elements of X_train and use the predicted value to
    # determine to add to the count
    for i in range(len(X_train)):
        if pred(np.hstack((np.ones(1, dtype = int), X_train[i])), w) != y_train[i]:
            count = count + 1
    
    # Calculate the average error based on the count above
    avg = count/(len(X_train))
            
    return avg
    
def confMatrix(X_train,y_train,w):
    
    # Initialize the matrix values
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0
    
    # Calculate how many points were correctly classified to be -1
    for i in range(len(X_train)):
        if (pred(np.hstack((np.ones(1, dtype = int), X_train[i])), w) == -1) and (y_train[i] == -1):
            class_1 = class_1 + 1
    
    # Calculate how many points were predicted to be classified as -1 but are actually +1
    for j in range(len(y_train)):
        if (pred(np.hstack((np.ones(1, dtype = int), X_train[j])), w) == -1) and (y_train[j] == 1) :
            class_2 = class_2 + 1
            
    # Calculate how many points were predicted to be classified as +1 but are actually -1
    for k in range(len(y_train)):
        if (pred(np.hstack((np.ones(1, dtype = int), X_train[k])), w) == 1) and (y_train[k] == -1) :
            class_3 = class_3 + 1
    
    # Calculate how many points were correctly classified to be +1
    for l in range(len(X_train)):
        if (pred(np.hstack((np.ones(1, dtype = int), X_train[l])), w) == 1) and (y_train[l] == 1):
            class_4 = class_4 + 1

        
    # Append the values to the 2x2 confusion matrix
    con = np.array([[class_1, class_2],
           [class_3, class_4]])
        
        
    return con
    

def pred(X_i,w):
    
    # If the dot product of the feature vector and w is strictly positive
    # classify as +1, otherwise classify as -1
    if np.dot(X_i, w) > 0:
        return 1
    elif np.dot(X_i, w) <= 0:
        return -1
    
        
def test_SciKit(X_train, X_test, Y_train, Y_test):
    
    # Initialize the perceptron
    pct = Perceptron()
    
    # Train the training data points
    pct.fit(X_train,Y_train)
    
    # Find the predicted values from the test data points
    pred_pct = pct.predict(X_test)
    
    # Return the confusion matrix calculated using the test outputs and predicted outputs
    return confusion_matrix(Y_test,pred_pct)


    
def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
                    
    #Testing Part 1a
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)
    
    #Testing Part 1b
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
# The performance of my implementation compared to the scikit-learn library has a level of error
# which is causing the resulting values in the matrices to have some disruptancies between them.
# This is due to the fact that in my implementation, I classify an output of zero as miscalssified, 
# However the scikit-learn library is most likely using a margin of error, and misclassifies points 
# within a small range from zero, which is why the confusion matrix is different than my implementation.
