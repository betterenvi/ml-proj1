# !/usr/bin/python
# coding=utf-8

import re
import numpy as np
import pandas as pd
import os
trainfile = "adult.data"
testfile = "adult.test"
FEATURES = 14
continuous_index = (0, 2, 4, 10, 11, 12)	# For continous data, we need to discretize it
bins = (12,10,8,12,12,12) # The bins of each continuous feature
train_data = []; train_ans = []
test_data = []; test_ans = []
test_res = [] # record test results
features_0 = [dict() for _ in range(FEATURES)]; features_1 = [dict() for _ in range(FEATURES)] # 0 for res <= 50k, 1 for res > 50k
train_0 = 0; train_1 = 0 # 0 for lines whose res <= 50k, 1 for res > 50k in train set
test_0 = 0; test_1 = 0 # in test set

def preprocessing():	# delete unknown lines and discretize continous features
	global train_data
	global test_data
	try:
		f = open(trainfile, "r")
	except:
		os.exit()
	data1 = f.readlines()
	size1 = len(data1)
	i = 0
	while i < size1: # delete unknown lines
		if data1[i].find("?") == -1:
			i += 1
		else:
			data1.pop(i)
			size1 -= 1
	f.close()
#	data1 = list(set(data1))
#	size1 = len(data1)
	
	try:
		f = open(testfile, "r")
	except:
		os.exit()
	data2 = f.readlines()
	data2.pop(0)	# first line of the train set is not data
	size2 = len(data2)
	i = 0
	while i < size2:
		if data2[i].find("?") == -1:
			i += 1
		else:
			data2.pop(i)
			size2 -= 1
	#data2 = list(set(data2))
	f.close()
	
	train_set = []
	test_set = []
	for line in data1:
		line = line.strip()
		feature_list = re.split(", ", line)
		train_set.append(feature_list)
		
	for line in data2:
		line = line.strip()
		feature_list = re.split(", ", line)
		test_set.append(feature_list)
 
	train_set.extend(test_set) # discretize two sets in one batch
	data_total = discretizeData(train_set, continuous_index, bins)	
	train_data = data_total[:size1]
	test_data = data_total[size1:]

def discretizeFeature(data_of_feature, bin_num):	
	return pd.cut(data_of_feature, bin_num)
	
def discretizeData(data, continue_feature_list, bins):
	for feature_i_index in range(len(continue_feature_list)):
		feature = continue_feature_list[feature_i_index]
		data_of_feature_i = []
		size = len(data)
		for i in range(size):
			data_of_feature_i.append(float(data[i][feature]))
		discretized_feature_i = discretizeFeature(data_of_feature_i,bins[feature_i_index])
		for i in range(size):
			data[i][feature] = str(discretized_feature_i[i])
			
	return data
	
def gen_clean_data():
	global test_data
	global test_ans
	global train_data
	global train_ans
	global train_0
	global train_1
	global test_0
	global test_1
	
	print( "Generating clean data...")
	preprocessing()
	
	size = len(train_data)
	for i in range(size):
		if train_data[i][-1][0] == ">":
			train_ans.append(1)
			train_1 += 1
		else:
			train_ans.append(0)
			train_0 += 1
	
	size = len(test_data)
	for i in range(size):
		if test_data[i][-1][0] == ">":
			test_ans.append(1)
			test_1 += 1
		else:
			test_ans.append(0)
			test_0 += 1
	
def train():
	global features_0
	global features_1
#	gen_data(trainfile, "train")
	print "Start training..."
	for i in range(len(train_data)):
		ele = train_data[i]
		ans = train_ans[i]
		for j in range(FEATURES):
			if ans == 0:
				if ele[j] not in features_0[j]:
					features_0[j][ele[j]] = 1
				else:
					features_0[j][ele[j]] += 1
			else:
				if ele[j] not in features_1[j]:
					features_1[j][ele[j]] = 1
				else:
					features_1[j][ele[j]] += 1
	
def test():
	global test_res
	train_p_0 = train_0 * 1.0 / (train_0 + train_1)
	train_p_1 = train_1 * 1.0 / (train_0 + train_1)
	
	print "Start testing..."
	for i in range(len(test_data)):
		ele = test_data[i]
		product_0 = 1.0
		product_1 = 1.0
		for j in range(FEATURES):
			# cal product_0
			if ele[j] not in features_0[j]:
				product_0 = 0.0
#				product_0 *= 1.0 / train_0
			else:
				product_0 *=  (features_0[j][ele[j]] * 1.0 / train_0)
			
			# cal product_1
			if ele[j] not in features_1[j]:
				product_1 = 0.0
#				product_1 *= 1.0 / train_1
			else:
				product_1 *= (features_1[j][ele[j]] * 1.0 / train_1)
		
		product_0 *= train_p_0
		product_1 *= train_p_1
		p_0 = product_0 / (product_0 + product_1)
		p_1 = product_1 / (product_0 + product_1)
		if p_1 > p_0:
			test_res.append(1)
		else:
			test_res.append(0)

def cal():	# Calculate Accuracy, Precision, Recall, F-score
	correct = 0
	cnt_1_1 = 0	# cnt_i_j means that you classify the item into class i, while it actually belongs to j
	cnt_1_0 = 0
	cnt_0_1 = 0
	for i in range(len(test_ans)):
		if test_res[i] == 1:
			if test_ans[i] == 1:
				cnt_1_1 += 1
				correct += 1
			else:
				cnt_1_0 += 1
		else:
			if test_ans[i] == 1:
				cnt_0_1 += 1
			else:
				correct += 1
	accuracy = correct * 1.0 / len(test_ans)
	precision = cnt_1_1 * 1.0 / (cnt_1_1 + cnt_1_0)
	recall = cnt_1_1 * 1.0 / (cnt_1_1 + cnt_0_1)
	print "Accuracy: " + "%.3f%%"%(accuracy * 100)
	print "Precision: " + "%.3f%%"%(precision * 100)
	print "Recall: " + "%.3f%%"%(recall * 100)
	print "F-score: " + "%.3f"%(2 * precision * recall / (precision + recall))
	
if __name__ == "__main__":
	gen_clean_data()
	train()
	test()
	cal()