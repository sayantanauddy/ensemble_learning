# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 01:05:15 2017

@author: sayantan
"""

import csv
import numpy as np
from operator import itemgetter
from cart import split_dataset, classify, gini, split

def read_training_leaf():
    """
    Reads the training data file and formats it into attributes and classes
    The Kaggle Leaf dataset is used here.
    [https://www.kaggle.com/c/leaf-classification/data]
    Input:
        None
    Output:
        (attributes, classes): list of attributes and classes
        attributes is a list of lists, where each row is a data point,
        classes contains the label for the corresponding attribute row
    """    
    # Path of the dataset
    dataset = '/home/sayantan/computing/datasets/leaf_classification/train.csv'

    # Read the csv file and skip the header row
    with open(dataset, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the header
        data_list = list(reader)
        
    # Split into Id, Classes and Attributes
    id_num = [r[0] for r in data_list]  # Not used anymore
    classes = [r[1] for r in data_list]
    attributes = [r[2:] for r in data_list]
    attributes = [[float(y) for y in x] for x in attributes]
    
    return (attributes, classes)
    
    
def read_test_leaf():
    """
    Reads the test data file and formats it into attributes and classes
    The Kaggle Leaf dataset is used here.
    [https://www.kaggle.com/c/leaf-classification/data]
    Input:
        None
    Output:
        (attributes, classes): list of attributes and classes
        attributes is a list of lists, where each row is a data point,
        classes contains the label for the corresponding attribute row
    """    
    # Path of the dataset
    dataset = '/home/sayantan/computing/datasets/leaf_classification/test.csv'

    # Read the csv file and skip the header row
    with open(dataset, 'rb') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip the header
        data_list = list(reader)
        
    # Split into Id, Classes and Attributes
    id_nums = [r[0] for r in data_list] 
    attributes = [r[1:] for r in data_list]
    attributes = [[float(y) for y in x] for x in attributes]
    
    return (attributes, id_nums)
    

def bootstrap(data, bootstrap_size=None):
    
    attributes = data[0]
    classes = data[1]    
    
    if bootstrap_size is None:
        bootstrap_size = len(attributes)
        
    # Generate indices of bootstrap sample
    bootstrap_indices = list(np.random.choice(len(attributes), bootstrap_size))
    
    # Generate the bootstrap data
    bootstrap_attributes = list(itemgetter(*bootstrap_indices)(attributes))
    bootstrap_classes = list(itemgetter(*bootstrap_indices)(classes))
    
    return (bootstrap_attributes, bootstrap_classes)
    

def random_cart(data, num_rand_attr, min_rows):
    """
    Creates a random tree. Almost identical to the CART algorithm. Only difference
    here is that while searching for the best split location, instead of considering
    all attributes, only a random selection of attributes is considered.
    Note:
        (i)    All attributes are assumed to be numeric
        (ii)   No pruning is performed
    Input:
        data: (attributes, classes) where attributes contains rows of attributes, and classes
        contains the label for the corresponding row.
        num_rand_attr: Number of randomly drawn attributes to be considered for a split
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
        
        # Instead of searching all attributes for the best split location, 
        # 'num_rand_attr' number of attributes are randomly chosen
        rand_attributes = np.random.choice(len(attributes[0]), size=num_rand_attr, replace=False)
        for attribute_number in rand_attributes:
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

        # In the rare case when a node has multile rows which are exact copies,
        # then a split location cannot be found even though the number of rows
        # is above the threshold. In this case, the entire node is labelled as 
        # a leaf.
        # If this is not done, then the recursive calls for split will continue
        # indefinitely (since best_attribute_index=-1 and best_attribute_split_val=-1)
        # and when the recursion limit is reached, a runtime exception will be thrown.
        # This situation does not always occur, but may result based on the way
        # the subsamples are created.

        if best_attribute_index == -1:
            root = { \
            'decision_attribute':None, \
            'decision_attribute_val':None, \
            'children':{}, \
            'label':'leaf', \
            'classes': classes}
                    
        else:    
            
            # The first set is where the value of the split attribute is <= best_attribute_split_val 
            split_data = split(attributes, classes, best_attribute_index, best_attribute_split_val)
            
            # Create a root node    
            # Node(parent,decision_attribute,label)
            # The key 'le' stands for less than or equal to. 'gt' => greather than
            root = {'decision_attribute':best_attribute_index, \
                    'decision_attribute_val':best_attribute_split_val,\
                    'children':{'le':random_cart(split_data[0], num_rand_attr, min_rows), 'gt':random_cart(split_data[1], num_rand_attr, min_rows)}, \
                    'label':None, \
                    'classes': None}
            
        
        return root    
    
    
def random_forest(training_data, num_trees, num_rand_attr, bootstrap_size=None):


    # Initialize the ensemble of trees
    ensemble = []
    
    # Random forest logic
    for i in range(num_trees):
        
        # Draw a bootstrap sample of size 'bootstrap_size'
        bootstrap_data = bootstrap(training_data, bootstrap_size)
        
        # Grow a random forest tree using the bootstrap sample and randomly 
        # drawn 'num_rand_attr' attributes to decide the split at each node
        tree = random_cart(bootstrap_data, num_rand_attr, 50)
        
        # Add the tree to the ensemble
        ensemble.append(tree)
        print "Number of trees: " + str(len(ensemble))

    return ensemble
    
    
def classify_kaggle(test_data, ensemble):
    """
    Outputs probability of each class for a row, as per the required format in Kaggle
    """
    attributes = test_data[0]
    id_nums = test_data[1]
    
    # Create an output file for the prediction results
    submission_file = open('/home/sayantan/computing/datasets/leaf_classification/submission.csv', 'w')    
     
    # Read the header row of the file 'sample_submission.csv' and extract all 
    # the class names
    with open('/home/sayantan/computing/datasets/leaf_classification/sample_submission.csv', 'r') as f:
        first_line = f.readline().strip()  
        
    # Remove 'id' from the first line and store the class names in a list
    all_classes = first_line.split(",")[1:]
    
    # Write the header line in the submission file
    submission_file.write("%s\n" %first_line)
    
    
    for index in range(len(attributes)):
        row_id = id_nums[index]
        row_attribute = attributes[index]
        ensemble_predictions = {}
        
        line_to_write = str(row_id)
        
        # Each tree makes a prediction for the current row
        for tree in ensemble:
            prediction  = classify(row_attribute, tree)
            if prediction in ensemble_predictions:
                ensemble_predictions[prediction] += 1.0/len(ensemble)
            else:
                ensemble_predictions[prediction] = 1.0/len(ensemble)
                
        # Iterate through all the class names. If a class name exists in the
        # predictions made by the ensemble append the predicted value to the
        # string to else, if not then append 0.0
        for class_name in all_classes:
            if class_name in ensemble_predictions.keys():
                line_to_write = line_to_write + "," + str(ensemble_predictions[class_name])
            else:
                line_to_write = line_to_write + ",0.0"
                
            print line_to_write
                
        # Write to the submission file
        submission_file.write("%s\n" %line_to_write)
        
    # Close the writer
    submission_file.close()
    
        
def main():
    """
    Main function.
    Divides the breast canser data set into training and testing set (80-20 ratio).
    Note:
        (i)    Here all attributes are numeric (integers)
        (ii)   Attributes with missing data are skipped (handled by readdata function)
    """
    # Get the training data
    training_data = read_training_leaf()
    num_attributes = len(training_data[0][0])
    
    # Create the random forest
    ensemble = random_forest(training_data, num_trees=100, num_rand_attr=15, bootstrap_size=None)
    
    print ensemble
    
    # Get the test data
    test_data = read_test_leaf()
    
    # For each row in test_data, each tree makes a classification decision.
    # The counts are then converted into probabilities by using the relative
    # class frequency
    classify_kaggle(test_data, ensemble)
    
        
if __name__ == "__main__":
    main()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
