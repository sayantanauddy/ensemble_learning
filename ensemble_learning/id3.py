# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:28:17 2017

@author: sayantan
Simple implementation of the ID3 algorithm. No pruning is performed and missing
attributes are not handled. A simple dataset is used, as shown in the main function.

Based on the ID3 algorithm in the following book:
'Machine Learning: An Algorithmic Perspective (2nd edition)' - Marsland
"""

from math import log
import operator


def uniquecounts(rows):
    """
    Returns the counts of possible results (the last column of each row is the result)
    Input:
        rows: list of data lists
    Output:
        Counts of possible results as a Dict (key=result, value=count)
    """
    unique_counts = {}
    for row in rows:
        if row[len(row)-1] in unique_counts:
            unique_counts[row[len(row)-1]] += 1
        else:
            unique_counts[row[len(row)-1]] = 1
    return unique_counts
    

def entropy(rows):
    """
    Entropy is the sum of -p(x)log(p(x)) across all the different possible results
    Input:
        rows: list of data lists
    Output:
        Entropy of the input data
    """
    unique_counts = uniquecounts(rows)
    ent = 0
    for item in unique_counts:
        prob_of_item = unique_counts[item]/float(sum(unique_counts.values()))
        ent += prob_of_item*log(prob_of_item,2)
        
    return -1.0*ent


def information_gain(rows, column):
    """
    Calculates the information gain when the set is divided based on the provided column
    Input:
        rows: list of data lists
        column: column number or attribute for which gain is required
    Output:
        Information gain of using 'column' as a decision attribute
    """
    # Calculate the entropy of the undivided set
    entropy_undivided = entropy(rows)
    
    # Compute the unique values and counts of the provided column (feature)
    unique_col_vals = {}
    for row in rows:
        if row[column] in unique_col_vals:
            unique_col_vals[row[column]] += 1
        else:
            unique_col_vals[row[column]] = 1
    
    summation_val = 0
    for col_val in unique_col_vals.keys():
        # Determine subset of 'rows' where row[column]==col_val
        subset = [row for row in rows if row[column] == col_val]
        summation_val += (unique_col_vals[col_val]/float(len(rows))*entropy(subset))
    
    # Calculate the gain and return it        
    gain = entropy_undivided - summation_val    
    return gain
    

def id3(rows, target_column, attributes, data):
    """
    Implementation of the ID3 algorithm
    Input:
        rows: list of data lists
        target_column: the column which holds the classification label
        attributes: set of columns to be considered as attributes
        data: complete data list ('rows' changes in recursive calls, 'data' doesn't)
    Output:
        Dict representing a node with the following structure:
        {'decision_attribute':dummy, 'children':{dummy_key:dummy_value}, 'label':dummy}
        'children' is a Dict of Dicts where the key is the value of the decision 
        attribute for the branch connected to that particular child node
    """
    
    # Create a root node    
    # Node(parent,decision_attribute,label)
    root = {'decision_attribute':None, 'children':{}, 'label':None}
    
    # Check if all rows have the same label. If yes, then set the label of root
    # to this label and return the tree with only root
    unique_data_labels = uniquecounts(rows)
    if len(unique_data_labels) == 1:
        root['label'] = unique_data_labels.keys()[0]
        # The tree is returned as a dict with the node as the key and a list of
        # children as the value for that key
        return root
    
    # If 'attributes' is empty then return root as the only node of the tree. 
    # Set the label of root as the most common label in the data
    if len(attributes) == 0:
        # Sort the labels in descending order of occurances
        sorted_data_labels = sorted(unique_data_labels.items(), key=operator.itemgetter(1), reverse=True)
        # Pick the first label in the list (most occurances)
        root['label'] = sorted_data_labels[0][0]
        return root
        
    # If the previous conditions are not satisfied
    
    # Pick the attribute from 'attributes' which has the highest information gain
    max_info_gain = 0
    best_attr = -1
    for column in attributes:
        info_gain = information_gain(rows, column)
        if info_gain>max_info_gain:
            best_attr = column
            max_info_gain = info_gain
    
    # Set the best attribute as the decision attribute of 'root'
    root['decision_attribute'] = best_attr
    
    # Find out the possible values for this best attribute
    poss_vals = list(set([r[best_attr] for r in data]))
    
    # For each possible value of attribute 'best_attr'
    for poss_val in poss_vals:
        # Find out subset of 'rows' where 'rows[best_attr]==poss_val
        rows_subset = [r for r in rows if r[best_attr]==poss_val]
    
        # If 'rows_subset' is empty then add a leaf node with label=most common
        # value of 'target_column' in rows
        if len(rows_subset) == 0:
            # Find the most common value of the target attribute (column) in 'rows'
            # and set it as the label of the leaf node
            
            # leaf node has no children
            # Node(parent,decision_attribute,label)
            leaf_node = {'decision_attribute':None, 'children':{}, 'label':sorted(unique_data_labels.items(), key=operator.itemgetter(1), reverse=True)[0][0]}
            # Return the tree comprising of the root node and the leaf node
            root['children'][poss_val] = leaf_node
            
        else:
            # Compute the subtree for the subset
            
            if len(attributes) <= 1:
                attributes = set()
            else:
                # Remove the best attribute from the set of attributes for the 
                # function call
                new_attributes = attributes.copy()
                new_attributes.remove(best_attr)
            
            subtree = id3(rows_subset,target_column,new_attributes,data)
            
            # Merge this subtree with 'root'
            # First retrieve the dict of children of root
            root['children'][poss_val] = subtree       
            
    return root
   

def classify(row, decision_tree):
    """
    Classifies a data row using the decision tree in a recursive manner
    Input:
        row: row of data to be classified
        decision_tree: the decision tree Dict
    Output:
        Classification result
    """
    if decision_tree['children'] == {}:
        return decision_tree['label']
    else:
        decision_attribute = decision_tree['decision_attribute']
        row_val_for_decision_attribute = row[decision_attribute]
        subtree = decision_tree['children'][row_val_for_decision_attribute]
        ret = classify(row, subtree)
        return ret
    

def main():
    """
    Main function
    Sets up data and calls the id3 function, followed by a call to print the tree
    """
    data=[['Sunny','Hot','High','Weak','No'],
    ['Sunny','Hot','High','Strong','No'],
    ['Overcast','Hot','High','Weak','Yes'],
    ['Rain','Mild','High','Weak','Yes'],
    ['Rain','Cool','Normal','Weak','Yes'],
    ['Rain','Cool','Normal','Strong','No'],
    ['Overcast','Cool','Normal','Strong','Yes'],
    ['Sunny','Mild','High','Weak','No'],
    ['Sunny','Cool','Normal','Weak','Yes'],
    ['Rain','Mild','Normal','Weak','Yes'],
    ['Sunny','Mild','Normal','Strong','Yes'],
    ['Overcast','Mild','High','Strong','Yes'],
    ['Overcast','Hot','Normal','Weak','Yes'],
    ['Rain','Mild','High','Strong','No']]

    dt = id3(data, 4, set([0,1,2,3]),data)
    print dt
    
 
if __name__ == "__main__":
    main()
