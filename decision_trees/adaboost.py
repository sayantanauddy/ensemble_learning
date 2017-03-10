# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 17:26:42 2017

@author: sayantan
Implementation of the Adaboost algorithm (classification with numeric 
attributes only). Both binary and multiclass classification is handled

This program is based on the following paper (for binary classification):
'AdaBoost and the Super Bowl of Classifiers: A Tutorial Introduction to 
Adaptive Boosting' - Rojas.

The following changes have been made to the algorithm in the above paper (for
handling multiclass classification):

    (1)    Labels are not converted to +1 or -1, they are kept as is
    (2)    The classification decision is not taken by the sign of C(x), but
           instead is taken to be the class that was given the maximum weight
           by the committee of classifiers
    (3)    Weights of the data points are normalized
    (4)    Multiclass Adaptive boosting is also handled
    (5)    Best classifier's weight is set as:
           best_classifier_weight = np.log((1-e_m)/e_m) + np.log(num_classes - 1)
           This handles both binary and multi-class classification
           
These changes are based on the following paper:
'Multi-class AdaBoost' - Zhu, Zou, Rosset, Hastie

The breast cancer wisconsin dataset has been used
[https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/]
"""

import numpy as np
import operator
from cart import split_dataset, cart, classify

def readdata_bc(datafile):
    """
    Reads the data file and formats it into attributes and classes
    Rows with missing attributes are skipped
    Input:
        datafile: string containing the full path of the data file
    Output:
        (attributes, classes): list of attributes and classes
        attributes is a list of lists, where each row is a data point,
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
                classes.append(line.split(',')[10])
                        
    return (attributes, classes)
    

def adaboost(training_data, pool_size, subsample_ratio, committee_size, num_classes):
    """
    Implementation of the Apative Boosting algorithm (binary and multiclass classification)
    Input:
        training_data: Training data consisting of (attributes, classes)
        where attributes is a 2D list with each data point in a row, classes
        is a 1D list containing the labels of the corresponding rows
        pool_size: Number of classifiers to be part of the pool
        subsample_ratio: The fraction of training data (randomly picked) that 
        should be used to train each weak classifier in the pool
        committee_size: Number of classifiers which take the final decision
        num_classes: Number of target classes
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
    
    # Normalize the weights
    weight_matrix_data = weight_matrix_data/sum(weight_matrix_data)
    
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
        best_classifier_weight = np.log((1-e_m)/e_m) + np.log(num_classes - 1)
        
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
                
        # Normalize the weight matrix
        weight_matrix_data = weight_matrix_data/sum(weight_matrix_data)
        
    # Now classification can be done by the committee of chosen classifiers
    return chosen_classifiers
            
            
def ensemble_classify(committee_of_classifiers, data_point):
    """
    Classifies a data point by a committee of classifiers
    Input:
        committee_of_classifiers: A list of classifiers, each element is a tuple
        (classifier, weight) where classifier is a CART tree (in the form of a dict)
        as used in cart.py and weight is a scalar. 
        data_point: Row of attributes
    Output:
        classification_label: The classification result
    """    
    classify_result_vote = {}
    for classifier in committee_of_classifiers:
        # Each classifier votes for a particular class
        # Weight of this vote = weight of the classifier
        label = classify(data_point,classifier[0])
        if label not in classify_result_vote.keys():
            classify_result_vote[label] = classifier[1]
        else:
            classify_result_vote[label] += classifier[1]
        
    # After voting is complete by the committee of classifiers, the class with the maximum
    # votes is taken to be the winner
    classification_label = max(classify_result_vote.iteritems(), key=operator.itemgetter(1))[0]
    return classification_label
    
    
def main():
    """
    Main function.
    Divides the breast canser data set into training and testing set (80-20 ratio).
    Note:
        (i)    Here all attributes are numeric (integers)
        (ii)   Attributes with missing data are skipped (handled by readdata function)
    """
    # Specify the data file
    datafile = '/home/sayantan/computing/datasets/uci_breastcanser_wisconsin_originial/breast-cancer-wisconsin.data'
    
    # Split the dataset into training and test sets in the ratio 80-20
    (training_data, test_data) = split_dataset(readdata_bc(datafile), 0.8)

    # Use the training data to train an ensemble of weighted classifiers
    committee_of_classifiers = adaboost(training_data, pool_size=50, subsample_ratio=0.5, committee_size=10, num_classes=2)    
    
    # Classify the test data using the weighted classifiers and calculate the error
    # Total error is just the 1-0 loss
    err_count = 0
    
    for i in range(len(test_data[0])):
        # Classify a row of test data
        classification_result = ensemble_classify(committee_of_classifiers, test_data[0][i])
        # Check the result      
        if test_data[1][i] != classification_result[0]:
            err_count += 1.0
            
    # Display the overall accuracy as a percentage        
    accuracy = 1.0 - float(err_count)/len(test_data[0])
    print str(accuracy*100) + '%'


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    