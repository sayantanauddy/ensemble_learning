# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:23:14 2017

@author: sayantan
Implementation of the CART algorithm, considering a simple scenario without
pruning, only numerical attributes, categorical class labels and without handling
missing attributes. The iris data set is used here.
[https://archive.ics.uci.edu/ml/machine-learning-databases/iris/]
"""

import numpy as np


def readdata(datafile):
    """
    Reads the data file and formats it into attributes and classes
    Input:
        datafile: string containing the full path of the data file
    Output:
        (attributes, classes): list of attributes and classes
    """
    
    attributes = []  # List for attributes (float)
    classes = []  # List for class categories (string)
    
    # Read the data file
    with open(datafile) as f:
        lines = f.read().splitlines()
        
    # Convert to appropriate type and store in 'attribute' and 'classes'
    for line in lines:
        if len(line)>0:
            attributes.append(map(float,line.split(',')[0:4]))
            classes.append(line.split(',')[4])

    return (attributes, classes)
    

def uniquecounts(classes):
    """
    Returns the counts of classes
    Input:
        classes: class categories
    Output:
        Counts of possible results as a Dict (key=result, value=count)
    """
    
    unique_counts = {}
    for row in classes:
        if row in unique_counts:
            unique_counts[row] += 1.0
        else:
            unique_counts[row] = 1.0
    return unique_counts
    
    
def gini(classes):
    """
    Calculates the GINI index for a set of classes
    Input:
        classes: 1D list of class labels
    Output:
        the GINI index value for the input set
    """

    unique_counts = uniquecounts(classes)    
    sum_p = 0.0
    for v in unique_counts.values():
        sum_p += (v/float(sum(unique_counts.values())))**2
    return 1.0-sum_p
    
def split(attributes, classes, split_attribute_index, split_attribute_val):
    """
    Splits the data (attributes and classes) into two sets based on the split
    attribute. Handles both numerical and discrete split attributes.
    Input:
        attributes: List of attributes (one row for each class, each row has
        multiple columns)
        classes: List of classes for the attributes (one for each row)
        split_attribute_index: Index of attribute on which to split
        split_attribute_val: Value used to decide split
    Output:
        split_data = ((attributes1, classes1),(attributes2, classes2))
        The input data is split into two classes and returned
    """
    attributes1 = []
    classes1 = []
    attributes2 = []
    classes2 = []
    if type(split_attribute_val) is float or type(split_attribute_val) is int:
        split_func = lambda x,y: y <= x
    else:
        split_func = lambda x,y: y == x
    
    for index in range(len(attributes)):
        if split_func(split_attribute_val, attributes[index][split_attribute_index]):
            attributes1.append(attributes[index])
            classes1.append(classes[index])
        else:
            attributes2.append(attributes[index])
            classes2.append(classes[index])
    
    return ((attributes1, classes1),(attributes2, classes2))


def cart(data, min_rows):
    """
        Implementation of the CART algorithm. 
        Note:
            (i)    All attributes are assumed to be numeric
            (ii)   No pruning is performed
        Input:
            data: (attributes, classes) where attributes contains rows of attributes, and classes
            contains the label for the corresponding row.
            min_rows: if the number of rows in a node is less or equal to this number,
            then that node is not split any further
        Output:
            root: A tree in the form of a dictionary. Root can have the following 
            format.
                    root = { \
                            'decision_attribute': <>, \
                            'decision_attribute_val': <>, \
                            'children': {'le': <left subtree>, 'gt': <right subtree>}, \
                            'label': <'leaf' for leaf node, else None>, \
                            'classes': <class labels if label='leaf'>}
    """
    # Decide split variable
    # All attributes are numerical
    
    # Outer loop - for all attributes
    # Inner loop - for all unique values of attributes
    attributes = data[0]
    classes = data[1]
    
    # First we need to decide if 'data' needs to be split further
    # If the number of rows in data is less than 5% of the training data
    # We do not split and instead return a node with all the classes in 'data'
    if len(attributes) <= min_rows:
        root = { \
            'decision_attribute':None, \
            'decision_attribute_val':None, \
            'children':{}, \
            'label':'leaf', \
            'classes': classes}
        return root
    else:
        # If the number of data points is above the threshold, these can be 
        # further split up
        lowest_gini_sum = 100
        best_attribute_index = -1
        best_attribute_split_val = -1
        for attribute_number in range(len(attributes[0])):
            # Get all unique values for a particular attribute and sort them
            all_values = list(set([row[attribute_number] for row in attributes]))
            all_values.sort()
            
            # If all_values contains n values, there are n-1 possible split locations
            # If x = [1,2,3,4,5] then zip(x,x[1::]) = [(1, 2), (2, 3), (3, 4), (4, 5)]
            # and [ (a+b)/2.0 for a,b in zip(x,x[1::])] = [1.5, 2.5, 3.5, 4.5]
            split_locs = [ (a+b)/2.0 for a,b in zip(all_values,all_values[1::])]
            
            # For each split_location find the sum of gini index of the resulting 2 classes
            for split_val in split_locs:
                split_data = split(attributes, classes, attribute_number, split_val)
                gini_sum = gini(split_data[0][1]) + gini(split_data[1][1])
                if gini_sum<lowest_gini_sum:
                    lowest_gini_sum = gini_sum
                    best_attribute_index = attribute_number
                    best_attribute_split_val = split_val

                    
        # The first set is where the value of the split attribute is <= best_attribute_split_val 
        split_data = split(attributes, classes, best_attribute_index, best_attribute_split_val)
        
        # Create a root node    
        # Node(parent,decision_attribute,label)
        # The key 'le' stands for less than or equal to. 'gt' => greather than
        root = {'decision_attribute':best_attribute_index, \
                'decision_attribute_val':best_attribute_split_val,\
                'children':{'le':cart(split_data[0], min_rows), 'gt':cart(split_data[1], min_rows)}, \
                'label':None, \
                'classes': None}
        
        return root


def classify(row, cart_tree):
    """
    Classifies a data row using the decision tree in a recursive manner
    Input:
        row: row of data to be classified
        decision_tree: the decision tree Dict
    Output:
        Classification result
    """
    if cart_tree['label'] == 'leaf':
        return cart_tree['classes']
    else:
        decision_attribute_index = cart_tree['decision_attribute']
        row_val_for_decision_attribute = row[decision_attribute_index]
        decision_attribute_val = cart_tree['decision_attribute_val']
        if row_val_for_decision_attribute <= decision_attribute_val:
            subtree = cart_tree['children']['le']
        else:
            subtree = cart_tree['children']['gt']
        ret = classify(row, subtree)
        return ret
        
    
        
    
def main():
    """
    Main function.
    Divides the iris data set into training and testing set (80-20 ratio).
    Note:
        (i)    Here all attributes are numeric
        (ii)   Here classification is performed but the same algorithm can be
               modified to perform regression. For regression, the standard deviation
               of a node can be used in place of the GINI index, and the classification
               function can be replaced by a function which returns the average of the
               final leaf node.
        (iii)  Missing attributes are not handled
    """
    # Specify the data file
    datafile = '/home/sayantan/computing/datasets/iris/iris.data'
    
    # Fetch the data
    data = readdata(datafile)
    
    # Divide the data into training set and test set
    # training = 80%, test = 20%    
    attributes = data[0]
    classes = data[1]
    total_row_count = len(attributes)
    training_row_count = int(round(0.8*total_row_count))
    
    # From 0 to (total_row_count-1) randomly pick training_row_count numbers
    training_row_indices = list(np.random.choice(range(total_row_count), training_row_count, replace=False))
    
    training_attributes = []
    training_classes = []
    test_attributes = []
    test_classes = []
    
    for i in range(total_row_count):
        if i in training_row_indices:
            training_attributes.append(attributes[i])
            training_classes.append(classes[i])
        else:
            test_attributes.append(attributes[i])
            test_classes.append(classes[i])
            
    training_data = (training_attributes, training_classes)
    
    # If number of rows in a node is <= this number then do not split further
    min_rows = int(round(0.05*total_row_count))
    
    # Training: Build the decision tree
    cart_tree = cart(training_data, min_rows)
    
    # Testing: Use the built tree to find the classification error on the test set
    # Total error is just the 1-0 loss
    err_count = 0
    for i in range(len(test_attributes)):
        if test_classes[i] not in classify(test_attributes[i], cart_tree):
            err_count += 1.0
            
    # Display the accuracy as a percentage        
    accuracy = 1.0 - float(err_count)/len(test_attributes)
    print str(accuracy*100) + '%'

if __name__ == "__main__":
    main()
    
    
    