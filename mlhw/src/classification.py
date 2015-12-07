"""
================================
Nearest Neighbors Classification
================================

"""

import sys
import math
import numpy as np
import random
from sklearn import neighbors, datasets, decomposition
from sklearn import cross_validation as cv

def get_best_n(input_features,input_labels):
    error_list = []
    limit = len(input_features)*0.7 if len(input_features)*0.7 < 50 else 50
    for n_neighbors in range(2,int(limit)):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        error = 1-cv.cross_val_score(clf,input_features,input_labels,cv=5).mean()
        error_list.append(error)   
     
    min_error = min(error_list)
    best_n_neighbors = error_list.index(min_error)+1
    return best_n_neighbors
    
if __name__=="__main__":

    testset_name=sys.argv[1]
    # import some data to play with

    feature_file =  open(('../data/{0}/TrainData.txt').format(testset_name))

    # extracting the values from the file for training data 
    input_features = []
    for feature in feature_file:
        temparray= np.array([float(x) for x in feature.split('\t')])
        input_features.append(temparray) 
    input_features = np.array(input_features)

    # extracting the values from the file for target data 
    target_file =  open(('../data/{0}/TrainLabel.txt').format(testset_name))
    input_labels = np.array([int(label.strip()) for label in target_file]) 

    
    test_feature_file =  open(('../data/{0}/TestData.txt').format(testset_name))

    # extracting the values from the file for testing data 

    test_features = []
    for test_feature in test_feature_file:
        temparray= np.array([float(x) for x in test_feature.split('\t')])
        test_features.append(temparray) 
    test_features = np.array(test_features)

    # avoid this ugly slicing by using a two-dim dataset


    spca = decomposition.KernelPCA(n_components=70,kernel="rbf",remove_zero_eig=True)
    input_features = spca.fit_transform(input_features)
    
    n_neighbors =   get_best_n(input_features,input_labels)
    
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(input_features, input_labels)

    test_features = spca.fit_transform(test_features)
    predicted_values = clf.predict(test_features)

    print predicted_values

