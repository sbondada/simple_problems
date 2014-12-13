"""
================================
Nearest Neighbors Classification
================================

"""

import sys
import math
import numpy as np
import random
from sklearn import neighbors, datasets

def calculate_error(actual_values,predicted_values):
    error_sq = 0
    rmse = 0
    for i in range(len(actual_values)):
        error_sq += math.pow(actual_values[i]-predicted_values[i],2)

    rmse = math.sqrt(error_sq/len(actual_values))
    #nrmse = rmse/(max(actual_values)-min(actual_values))
    return rmse

def get_features_label_from_idxs(total_feature_list,total_label_list,idx_list):
    partial_feature_list = []
    partial_label_list = []
    for idx in idx_list: 
        partial_feature_list.append(total_feature_list[idx])
        partial_label_list.append(total_label_list[idx])
    return(np.array(partial_feature_list),np.array(partial_label_list))

def generate_train_and_validation_set(input_features,
                                      input_labels,
                                      validation_set_percent):
    noofiterations = int(validation_set_percent*len(input_features))
    train_idxs = range(len(input_features))    
    validation_idxs = []
    for i in range(noofiterations):
        randpos = random.randint(0,len(train_idxs)-1) 
        validation_idxs.append(train_idxs.pop(randpos))
    return (get_features_label_from_idxs(input_features,
                                         input_labels,
                                         train_idxs),
            get_features_label_from_idxs(input_features,
                                         input_labels,
                                         validation_idxs))
    

def check_for_best_n(input_features,input_labels):
    training_set, validation_set = generate_train_and_validation_set(input_features,
                                                                     input_labels,
                                                                     0.20)

    # we create an instance of Neighbours Classifier and fit the data.
    error_list = []
    for n_neighbors in range(1,50):
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
        clf.fit(training_set[0],training_set[1])

        predicted_target = clf.predict(validation_set[0])
        error = calculate_error(validation_set[1],predicted_target)
        error_list.append(error)   
        print error
        print predicted_target
        print validation_set[1]
     
    min_error = min(error_list)
    print min_error
    best_n_neighbors = error_list.index(min_error)+1
    print best_n_neighbors
    
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

    n_neighbors = 3

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(input_features, input_labels)

    predicted_values = clf.predict(test_features)

    print predicted_values

