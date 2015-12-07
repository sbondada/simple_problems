"""
================================
missing value estimation
===============================
"""

import sys
import numpy as np
from sklearn import svm

def seperate_missing_from_full(missingdata_file):
    missing_dataset = []
    full_dataset = []
    for feature in missingdata_file:
        temparray= [float(x) for x in feature.split('\t')]
        if float(1.00000000000000e+99) in temparray:
            missing_dataset.append(temparray)
        else:
            full_dataset.append(temparray)
    return (full_dataset,missing_dataset)

def predict_missing_value(full_dataset,missingdata_feature):
    itemindex = np.where(np.array(missingdata_feature)==float(1.00000000000000e+99))
    index_list = sorted(itemindex[0].tolist(),reverse=True)    
    for idx in index_list:
        missingdata_feature.pop(idx)
    feature_set,label_set_by_idx = get_feature_and_label_set(full_dataset,index_list)
    clf = svm.SVR()
    idx_count = 0
    predicted_values = []
    for idx in index_list:
       clf.fit(feature_set,label_set_by_idx[idx_count]) 
       predicted_values.append(clf.predict(np.array([missingdata_feature]))[0])
       idx_count+=1
    return (predicted_values,index_list)

def get_feature_and_label_set(full_dataset,missingdata_idx_list):
   feature_set = []
   label_set = []
   temparray=[]
   for i in range(len(missingdata_idx_list)):
           temparray.append([])
   for record in full_dataset:
       idx_count = 0
       # keep the orginal list intact
       record = list(record)
       for idx in missingdata_idx_list:
            temparray[idx_count].append(record[idx])
            # this will work because the missingdata_idx_list is always going to
            # be in increasing order
            record.pop(idx)
            idx_count+=1
       feature_set.append(record)
   label_set_by_idx = temparray
   return (feature_set,label_set_by_idx)
        

if __name__=="__main__":

    testset_name=sys.argv[1]
    missingdata_file = open(('../data/{0}/MissingData.txt').format(testset_name)) 

    #extrating the values in the file
    full_dataset,missing_dataset = seperate_missing_from_full(missingdata_file)
    #reopening the decriptor as it has been already in use
    missingdata_file = open(('../data/{0}/MissingData.txt').format(testset_name)) 
    for feature in missingdata_file:
        temparray= [float(x) for x in feature.split('\t')]
        if float(1.00000000000000e+99) in temparray:
            predicted_values,idx_list = predict_missing_value(full_dataset,list(temparray))
            idx_count=0
            for idx in idx_list:
                temparray[idx] = predicted_values[idx_count]
                idx_count+=1
        print temparray 
    
