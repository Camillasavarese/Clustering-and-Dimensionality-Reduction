# Clustering-and-Dimensionality-Reduction
 
 ## Goal

The Goal of thisproject is to apply a dimensionality reduction on a big dataset in order to remove noise and then to apply the kmenas algorithm to divide the songs in clusters  and try to understand the results using pivotal tables.

This repository contains:
- main.py: this is the file with our results and comments of exercise 1, 2 and the algorithmic question 3
- utilities.py: module containing functions to extract the audio peaks from songs and other useful functions
- first.py: module containing functions to create elements for LSH algorithm (shingles, permutations, signatures, buckets)
- queries.py: module containing functions to make shingles, permutations, signatures and buckets to retrieve information and finalize the matching process and function used in the alternative LSH algorithm.
- ourKmeans.py: module containing two functions, the first one computes the KMeans algorithm and the second one does the gap statistics, to choose the best K value.
