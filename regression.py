import numpy as np
import sys
import math
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures

def read_inputs() :

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/age.csv'
	# raw_data = open(filename, 'rt')
	# age = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/fiq.csv'
	# raw_data = open(filename, 'rt')
	# fiq = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/sex.csv'
	# raw_data = open(filename, 'rt')
	# sex = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/label.csv'
	# raw_data = open(filename, 'rt')
	# label = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/hand.csv'
	# raw_data = open(filename, 'rt')
	# rhand = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# hand = np.zeros(shape=(497,3))

	# for i in range(0,497):
	# 	if rhand[i][0]==0:
	# 		hand[i][0]=1
	# 		hand[i][1]=0
	# 		hand[i][2]=0

	# 	if rhand[i][0]==1:
	# 		hand[i][0]=0
	# 		hand[i][1]=1
	# 		hand[i][2]=0

	# 	if rhand[i][0]==2:
	# 		hand[i][0]=0
	# 		hand[i][1]=0
	# 		hand[i][2]=1

	# # filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/dsm.csv'
	# # raw_data = open(filename, 'rt')
	# # rdsm = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# # dsm = np.zeros(shape=(497,5))

	# # for i in range(0,497):
	# # 	if rdsm[i][0]==0:
	# # 		dsm[i][0]=1
	# # 		dsm[i][1]=0
	# # 		dsm[i][2]=0
	# # 		dsm[i][3]=0
	# # 		dsm[i][4]=0

	# # 	if rdsm[i][0]==1:
	# # 		dsm[i][0]=0
	# # 		dsm[i][1]=1
	# # 		dsm[i][2]=0
	# # 		dsm[i][3]=0
	# # 		dsm[i][4]=0

	# # 	if rdsm[i][0]==2:
	# # 		dsm[i][0]=0
	# # 		dsm[i][1]=0
	# # 		dsm[i][2]=1
	# # 		dsm[i][3]=0
	# # 		dsm[i][4]=0

	# # 	if rdsm[i][0]==3:
	# # 		dsm[i][0]=0
	# # 		dsm[i][1]=0
	# # 		dsm[i][2]=0
	# # 		dsm[i][3]=1
	# # 		dsm[i][4]=0

	# # 	if rdsm[i][0]==4:
	# # 		dsm[i][0]=0
	# # 		dsm[i][1]=0
	# # 		dsm[i][2]=0
	# # 		dsm[i][3]=0
	# # 		dsm[i][4]=1

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/piq.csv'
	# raw_data = open(filename, 'rt')
	# piq = np.loadtxt(raw_data, delimiter=",").reshape((497,1))

	# filename = '/Users/Samir/documents/autproject_server_files/autism_prediction/regression_data/reg_data2/viq.csv'
	# raw_data = open(filename, 'rt')
	# viq = np.loadtxt(raw_data, delimiter=",").reshape((497,1))
	# feature_vec = np.hstack((age,sex,hand,fiq,piq,viq))

	# train_feature= feature_vec[0:440,0:8]
	# train_label = label[0:440,0]
	# test_feature= feature_vec[440:497,0:8]
	# test_label= label[440:497,0]
	# np.save('train_feature', train_feature, allow_pickle=True, fix_imports=True)
	# np.save('train_label', train_label, allow_pickle=True, fix_imports=True)
	# np.save('test_feature', test_feature, allow_pickle=True, fix_imports=True)
	# np.save('test_label', test_label, allow_pickle=True, fix_imports=True)

	
	feature_vec_train=np.load('train_feature.npy')
	#print(feature_vec_train.shape)
	train_label=np.load('train_label.npy')
	#print(train_label.shape)
	feature_vec_test=np.load('test_feature.npy')
	#print(feature_vec_test.shape)
	test_label=np.load('test_label.npy')
	#print(test_label.shape)
	return feature_vec_train,train_label,feature_vec_test,test_label

def accuracy(predictions,label,rows):

	count = 0
	predictions = np.rint(predictions)
	predictions.reshape(rows,1)
	label.reshape(rows,1)
	predictions[predictions>=1] = 1.0
	predictions[predictions<=0] = 0.0


	return accuracy_score(label,predictions)

def regression(feature_vec_train,train_label,feature_vec_test,test_label,a):
	reg = linear_model.Lasso(alpha = a)
	reg.fit(feature_vec_train,train_label)

	train_predictions = reg.predict(feature_vec_train)
	train_predictions.reshape(440,1)
	train_label.reshape(440,1)
	train_accuracy = accuracy(train_predictions,train_label,440)
	print("Train Accuracy: ")
	print(train_accuracy)

	test_predictions = reg.predict(feature_vec_test).reshape(57,1)
	test_predictions.reshape(57,1)
	test_label.reshape(57,1)
	test_accuracy = accuracy(test_predictions,test_label,57)

	print("Test Accuracy: ")
	print(test_accuracy)



if __name__ == "__main__":
	# read_inputs()
	feature_vec_train,train_label,feature_vec_test,test_label = read_inputs()
	print("Linear Regression")
	regression(feature_vec_train,train_label,feature_vec_test,test_label,0.1)

	print("Polynomial Regression")
	poly = PolynomialFeatures(degree=12)
	feature_vec_train = poly.fit_transform(feature_vec_train)
	feature_vec_test = poly.fit_transform(feature_vec_test)
	regression(feature_vec_train,train_label,feature_vec_test,test_label,0.5)


	



