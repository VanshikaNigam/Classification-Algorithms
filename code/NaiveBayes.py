import pandas as pd
from sklearn import preprocessing
import numpy as np
from collections import Counter
import math
from sklearn.utils import shuffle


file=pd.read_csv('project3_dataset1.txt',sep='\t',header=None)
datamatrix=file.as_matrix()
rows=datamatrix.shape[0]
col=datamatrix.shape[1]
del_col=0
k_folds=10
fold_size=math.ceil(rows/10)

for j in range(0,col):
	if datamatrix[0][j]=="Present" or datamatrix[0][j]=="Absent":
		del_col=j


def precompute_mean(data):
	mean_of_features=[]
	if del_col==0:
		for i in range(0,col-1):
			mean=np.mean(data[:,i])
			mean_of_features.append(mean)
	else:
		for i in range(0,del_col):
			mean=np.mean(data[:,i])
			mean_of_features.append(mean)
		mean_of_features.append(1)
		for i in range(del_col+1,col-1):
		
			mean=np.mean(data[:,i])
			mean_of_features.append(mean)
	return mean_of_features


def precompute_std(data,avg):
	std_of_features=[]
	if del_col==0:
		for i in range(0,col-1):
			
			variance=np.sum([np.power(x-avg[i],2) for x in data[:,i]])/len(data)
			std_of_features.append(math.sqrt(variance))
	else:
		for i in range(0,del_col):
			variance=np.sum([np.power(x-avg[i],2) for x in data[:,i]])/len(data)
			std_of_features.append(math.sqrt(variance))
		std_of_features.append(1)
		for i in range(del_col+1,col-1):
			variance=np.sum([np.power(x-avg[i],2) for x in data[:,i]])/len(data)
			std_of_features.append(math.sqrt(variance))
	return std_of_features


def calc_posterior(mean,std,data):
	posterior=[]
	
	if del_col==0:
		for i in range(0,len(data)-1):
			exp=np.exp(-np.power((data[i]-mean[i]),2)/(2*np.power(std[i],2)))
			base=1/(np.sqrt(2*np.pi)*std[i])
			posterior.append(base*exp)
			
	else:
		for i in range(0,del_col):
			exp=np.exp(-np.power((data[i]-mean[i]),2)/(2*np.power(std[i],2)))
			base=1/(np.sqrt(2*np.pi)*std[i])
			posterior.append(base*exp)
		posterior.append(1)

		for i in range(del_col+1,col-1):
			exp=np.exp(-np.power((data[i]-mean[i]),2)/(2*np.power(std[i],2)))
			base=1/(np.sqrt(2*np.pi)*std[i])
			posterior.append(base*exp)
	return posterior


def accuracy_cal(test_label,original_label):

	true_positive=0
	true_negative=0
	false_positive=0
	false_negative=0
	answer_list=[]
	
	for i in range(0,len(original_label)):
		if original_label[i]==1 and test_label[i]==1:
			true_positive=true_positive+1
		if original_label[i]==0 and test_label[i]==0:
			true_negative=true_negative+1
		if original_label[i]==0 and test_label[i]==1:
			false_positive=false_positive+1
		if original_label[i]==1 and test_label[i]==0:
			false_negative=false_negative+1

	if (true_positive+true_negative+false_negative+false_positive)>0:
		accuracy=(true_negative+true_positive)/(true_positive+true_negative+false_negative+false_positive)
	else:
		accuracy=0
	if (true_positive+false_positive)>0:
		precision=true_positive/(true_positive+false_positive)
	else:
		precision=0
	if (true_positive+false_negative)>0:
		recall=true_positive/(true_positive+false_negative)
	else:
		recall=0
	if (recall+precision)>0:
		F=(2*recall*precision)/(recall+precision)
	else:
		F=0
	answer_list.append(accuracy)
	answer_list.append(precision)
	answer_list.append(recall)
	answer_list.append(F)
	
	return answer_list





def naive_bayes(train,test):
	class0=[]
	class1=[]
	for i in range(0,len(train)):
		if train[i][col-1]==0:
			class0.append(train[i])
		elif train[i][col-1]==1:
			class1.append(train[i])
	class0=np.asmatrix(class0)
	class1=np.asmatrix(class1)
	
	prior0=class0.shape[0]/(train.shape[0])
	prior1=class1.shape[0]/(train.shape[0])
	mean_0=precompute_mean(class0)
	mean_1=precompute_mean(class1)

	std_0=precompute_std(class0,mean_0)
	std_1=precompute_std(class1,mean_1)
	str0=0
	str1=0
	s0=0
	s1=0
	
	if del_col>0:
		
		for i in range(0,len(class0)):
			if class0[i,del_col]=="Absent":
				str0=str0+1
			elif class0[i,del_col]=="Present":
				str1=str1+1
		str0_prob0=str0/(str0+str1)
		str1_prob0=str1/(str0+str1)
		
		for i in range(0,len(class1)):
			if class1[i,del_col]=="Absent":
				s0=s0+1
			elif class1[i,del_col]=="Present":
				s1=s1+1
		str0_prob1=s0/(s0+s1)
		str1_prob1=s1/(s0+s1)

	#print(str0_prob0, str1_prob0)

	test_label=[]
	original_label=[]
	for i in range(0,len(test)):
		#print(i,test[i])
		posterior_probability0=calc_posterior(mean_0,std_0,test[i])
		posterior_probability1=calc_posterior(mean_1,std_1,test[i])
		posterior0=np.prod(posterior_probability0)
		posterior1=np.prod(posterior_probability1)
		if del_col!=0:
			if test[i][del_col]=="Absent":
				class_posterior0=posterior0*prior0*str0_prob0
				class_posterior1=posterior1*prior1*str0_prob1
			elif test[i][del_col]=="Present":
				class_posterior0=posterior0*prior0*str1_prob0
				class_posterior1=posterior1*prior1*str1_prob1
		else:
			class_posterior0=posterior0*prior0
			class_posterior1=posterior1*prior1
		original_label.append(test[i][col-1])
		if class_posterior0>class_posterior1:
			test_label.append(0)
		else:
			test_label.append(1)
	
	answer_list=accuracy_cal(test_label,original_label)
	return answer_list




def k_fold(datamatrix,fold_size):

	
	temp_accuracy=0
	precision=0
	recall=0
	F=0
	np.random.shuffle(datamatrix)
	
	shuffle(datamatrix,random_state=0,n_samples=rows)

	answer_list=[]

	for i in range(0,k_folds):
		
		test_data=datamatrix[i*fold_size:i*fold_size+fold_size,:]
		train_data=np.delete(datamatrix,np.s_[i*fold_size:i*fold_size+fold_size],0)
		answer_list=naive_bayes(train_data,test_data)
		temp_accuracy=temp_accuracy+answer_list[0]
		precision=precision+answer_list[1]
		recall=recall+answer_list[2]
		F=F+answer_list[3]

		

	final_accuracy=temp_accuracy/(k_folds)
	final_recall=recall/k_folds
	final_precision=precision/k_folds
	final_F=F/k_folds
	print("final accuracy",final_accuracy)
	print("final recall",final_recall)
	print("final precision",final_precision)
	print("final f",final_F)

		


k_fold(datamatrix,fold_size)
