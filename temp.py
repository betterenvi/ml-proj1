# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import time    
from sklearn import metrics      
import pandas as pd  
from sklearn import preprocessing 



def prepareData(training_data_file_path,testing_data_file_path):
    #To split continues data and have the same split standard, we have to discretize the 
    #training data and testing data in one batch. And use the panda.cut to discretize .
        
    continue_feature_list = [0,2,4,10,11,12] #The index of continues feature
    bins = [10,12,8,12,12,12] #The bins of each continues feature
    data = []
    training_size = 0
    with open(training_data_file_path) as f:
        dataList = f.read().splitlines()    
    for datai in dataList:
        training_size += 1
        datai_feature_list = datai.split(", ")
        data.append(np.array(datai_feature_list))
        
    with open(testing_data_file_path) as f:
        dataList = f.read().splitlines()    
    for datai in dataList:
        datai_feature_list = datai.split(", ")
        data.append(np.array(datai_feature_list))
    data = np.array(data)
    discretizedData = discretizeData(data, continue_feature_list, bins)

    #return training data and testing data
    print ("training_size: ", training_size)
    return discretizedData[0:training_size,:],discretizedData[training_size:,:]

#data_of_feature:np.array, the data of a feature
#bin_num: to discretize to how many bins
def discretizeFeature(data_of_feature,bin_num):    
    return pd.cut(data_of_feature,bin_num)
    
    #data: np.ndarray, the training data
    #continue_attr: list
    #bins: the length of each discretized feature
    #To discretize the continues attribute/feature of data
def discretizeData(data,continue_feature_list,bins):
    for feature_i_index in range(len(continue_feature_list)):
        feature = continue_feature_list[feature_i_index]
        data_of_feature_i = np.array( [float(rowi) for rowi in data[:,feature] ] )#str to float
        discretized_feature_i = discretizeFeature(data_of_feature_i,bins[feature_i_index])
        print( discretized_feature_i)
        data[:,feature] = np.array(discretized_feature_i) #Use the discretized feature replace the continues feature
    return data
    
# Decision Tree Classifier    
def decision_tree_classifier(train_x, train_y):    
    from sklearn import tree    
    model = tree.DecisionTreeClassifier()    
    model.fit(train_x, train_y)    
    return model 
 
# Multinomial Naive Bayes Classifier    
def naive_bayes_classifier(train_x, train_y):    
    from sklearn.naive_bayes import MultinomialNB    
    model = MultinomialNB(alpha=0.01)    
    model.fit(train_x, train_y)    
    return model    
        
def main():

    training_data_file_path = "adult.data"
    testing_data_file_path = "adult.test"
    train, test = prepareData(training_data_file_path,testing_data_file_path)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for row in train:
        train_y.append(row[len(row)-1])
        for i in range(1,len(row)-1):
            train_x.append(row[i])
            
    for row in test:
        test_y.append(row[len(row)-1])
        for i in range(1,len(row)-1):
            test_x.append(row[i])
        
    test_classifiers = ['NB', 'DT']    
    classifiers = {'NB':naive_bayes_classifier, 'DT':decision_tree_classifier}    
    for classifier in test_classifiers:    
        print('******************* %s ********************' % classifier)    
        start_time = time.time()    
        model = classifiers[classifier](train_x, train_y)    
        print('training took %fs!' % (time.time() - start_time))    
        predict = model.predict(test_x)    
        precision = metrics.precision_score(test_y, predict)    
        recall = metrics.recall_score(test_y, predict)    
        print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))    
        accuracy = metrics.accuracy_score(test_y, predict)    
        print('accuracy: %.2f%%' % (100 * accuracy))     





