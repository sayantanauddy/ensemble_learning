# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:51:04 2017

@author: sayantan
"""
from math import log

# Divides a set on a specific column. Can handle numeric or nominal values
def divideset(rows, column, value):
    # Make a function that tells us if a row is in the first group (true) or 
    # the second group (false)
    split = None
    if isinstance(value,int) or isinstance(value,float):
        split = lambda x: x>=value
    else:
        split = lambda x: x==value
        
    set1 = [row for row in rows if split(row[column])]
    set2 = [row for row in rows if not split(row[column])]
    return (set1, set2)
    


# Create counts of possible results (the last column of each row is the result)
def uniquecounts(rows):
    unique_counts = {}
    for row in rows:
        if row[len(row)-1] in unique_counts:
            unique_counts[row[len(row)-1]] += 1
        else:
            unique_counts[row[len(row)-1]] = 1
    return unique_counts


# Entropy is the sum of -p(x)log(p(x)) across all 
# the different possible results
def entropy(rows):
    unique_counts = uniquecounts(rows)
    ent = 0
    for item in unique_counts:
        prob_of_item = unique_counts[item]/float(sum(unique_counts.values()))
        ent += prob_of_item*log(prob_of_item,2)
        
    return -1.0*ent

# Calculates the information gain when the set is divided based on the provided
# column
def information_gain(rows, column):
    
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
    
