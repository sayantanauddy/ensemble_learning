# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:26:42 2017

@author: sayantan
Implementation of the Adaboost algorithm (binary classification with numeric 
attributes). This program is based on 'AdaBoost and the Super Bowl of 
Classifiers: A Tutorial Introduction to Adaptive Boosting' by Rojas.
The breast cancer wisconsin dataset has been used
[https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/]
"""

import numpy as np
from cart import split_dataset, cart, classify

def readdata_bc(datafile):
    """
    Reads the data file and formats it into attributes and classes
    The dataset at https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
    is used.
    Last column is the label which is denoted by  4: malignant and 2: benign
    This is changed to 1: malignant and -1: benign
    Input:
        datafile: string containing the full path of the data file
    Output:
        (attributes, classes): list of attributes and classes
        attributes is a list of lists, where each row is a data point
        classes contains the label for the corresponding attribute row
    """
    
    attributes = []  # List for attributes (float)
    classes = []  # List for class categories (string)
    
    # Read the data file
    with open(datafile) as f:
        lines = f.read().splitlines()
        
    # Convert to appropriate type and store in 'attribute' and 'classes'
    for line in lines:
        if len(line)>0:
            line.replace(" ","")
            # Only consider data points which do not have any missing attributes
            if '?' not in line:
                attributes.append(map(float,line.split(',')[1:9]))
                if line.split(',')[10]=='2':
                    classes.append(-1)
                elif line.split(',')[10]=='4':
                    classes.append(1)
                        
    return (attributes, classes)
    

def adaboost(training_data, pool_size, subsample_ratio, committee_size):
    """
    Implementation of the Apative Boosting algorithm (based on )
    Input:
        training_data: Training data consisting of (attributes, classes)
        where attributes is a 2D list with each data point in a row, classes
        is a 1D list containing the labels of the corresponding rows
        pool_size: Number of classifiers to be part of the pool
        subsample_ratio: The ratio of training data (randomly picked) that 
        should be used to train each weak classifier in the pool
        committee_size: Number of classifiers which take the final decision
    Output:
        chosen_classifiers: List of classifiers where each element is a 
        tuple (classifier, weight_of_classifier). Here classifier is a CART tree
        (see cart.py for tree structure) and weight_of_classifier is the 
        importance of this classifier towards the final classification decision
    """

    # Build a pool of pool_size classifiers by randomly subsampling from the training data
    # Subsampling is done without replacement and each sub sample has subsample_ratio part
    # of the training data
    classifier_pool = []
    for i in range(pool_size):
        # To create the subsample, the training data is split in a 40-60 ratio
        # The second dataset which is returned is discarded
        (subsample_train_data, _) = split_dataset(training_data, subsample_ratio)
        
        # Build a cart tree from this subsample and append to classifier_pool
        min_rows = int(round(0.05*len(subsample_train_data[0])))
        classifier_pool.append(cart(subsample_train_data, min_rows))
        
    # Now consider the entire training set again
    # The real logic for adaboost starts here

    # Create the scouting matrix of size (number of data points)X(number or classifiers)
    # cell(i,j) of the scouting matrix = 0 if data_i is correctly classified by classifier_j
    # else it is = 1

    # Initialize the scouting_matrix
    scouting_matrix = np.zeros((len(training_data[0]), len(classifier_pool)), dtype=np.int)
    
    # Populate the scounting_matrix
    for i in range(len(training_data[0])):
        for j in range(len(classifier_pool)):
            if classify(training_data[0][i], classifier_pool[j]) != training_data[1][i]:
                scouting_matrix[i][j] = 1

    # Initialize the weight matrix for the data points
    weight_matrix_data = np.ones((len(training_data[0]),1))
    
    # Initialize a list to hold chosen classifiers
    # This list holds tuples of (classifier, weight)
    chosen_classifiers = []
    
    for m in range(committee_size):
        # Determine the error for each classifier
        # This can be done by multiplying transpose(weight_matrix_data) with scouting_matrix
        classifier_error = np.dot(np.transpose(weight_matrix_data),scouting_matrix)
        
        # Extract the classifier with the lowest error
        best_classifier_index = np.argmin(classifier_error)
        
        # Calculate the sum of weights of all data points
        W = np.sum(weight_matrix_data)
        
        # Calculate the sum of errors made by best classifier
        W_e = classifier_error[0, best_classifier_index]
        e_m = W_e/W
        
        # Calculate the weight of this classifier towards the final decision
        best_classifier_weight = 0.5*np.log((1-e_m)/e_m)
        
        # Add this classifier to the list of chosen classifiers
        chosen_classifiers.append((classifier_pool[best_classifier_index], best_classifier_weight))
        
        # Update the weight of the data points for the next iteration
        # First get the column from the scouting matrix for the best classifier
        best_classifier_scouting_col = scouting_matrix[:, best_classifier_index]
        
        for index in range(len(best_classifier_scouting_col)):
            if best_classifier_scouting_col[index] == 1:
                # For a miss
                weight_matrix_data[index,0] =  weight_matrix_data[index,0] * np.sqrt((1-e_m)/e_m)
            else:
                weight_matrix_data[index,0] =  weight_matrix_data[index,0] * np.sqrt(e_m/(1-e_m))                

        # Now classification can be done by the committee of chosen classifiers
        return chosen_classifiers
            

def main():
    """
    Main function.
    Divides the breast canser data set into training and testing set (80-20 ratio).
    Note:
        (i)    Here all attributes are numeric (integers)
        (ii)   Only binary classification is handled here
        (iii)  Labels are +1 for malignant and -1 for benign (handled by readdata function)
        (iv)   Attributes with missing data are skipped (handled by readdata function)
    """
    # Specify the data file
    datafile = '/home/sayantan/computing/datasets/uci_breastcanser_wisconsin_originial/breast-cancer-wisconsin.data'
    
    # Split the dataset into training and test in the ratio 80-20
    (training_data, test_data) = split_dataset(readdata_bc(datafile), 0.8)

    # Use the training data to get an ensemble of weighted classifiers
    chosen_classifiers = adaboost(training_data, pool_size=50, subsample_ratio=0.5, committee_size=10)    
    
    # Classify the test data using the weighted classifiers and calculate the error
    # Total error is just the 1-0 loss
    err_count = 0
    for i in range(len(test_data[0])):
        cumulative_sum = 0
        for classifier in chosen_classifiers:
            # cumulative_sum = wt1*classify(row, classifier1) + ....+ wtm*classify(row, classifierm)
            cumulative_sum += classifier[1]*classify(test_data[0][i],classifier[0])
            
        classification_result = np.sign(cumulative_sum)
            
        if test_data[1][i] != classification_result:
            err_count += 1.0
            
    # Display the accuracy as a percentage        
    accuracy = 1.0 - float(err_count)/len(test_data[0])
    print str(accuracy*100) + '%'


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    