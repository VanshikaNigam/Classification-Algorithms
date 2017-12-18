import pandas as pd
from sklearn import preprocessing
import numpy as np
from collections import Counter
import math
from sklearn.utils import shuffle


#load the input file
file=pd.read_csv('project3_dataset2.txt',sep='\t',header=None)
datamatrix=file.as_matrix()
rows=datamatrix.shape[0]
col=datamatrix.shape[1]

del_col=0
#to store the column
for i in range(0,col):
	
	if datamatrix[0][i]=="Present" or datamatrix[0][i]=="Absent":

		del_col=i

def handle_categorical_data(file):
	columns=file.columns.values

	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val]

		if file[column].dtype!=np.int64 and file[column].dtype!=np.float64:
			column_contents=file[column].values.tolist()
			uniques=set(column_contents)
			x=0
			for unique in uniques:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1
			file[column]=list(map(convert_to_int,file[column]))
	return file

datamatrix=handle_categorical_data(file).as_matrix()

train_data=[]
test_data=[]
k_folds=10
fold_size=math.ceil(rows/10)
k=5
train_label=[]
y_predict=0



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
	#accuracy=accuracy*100
	return answer_list
	#print("accuracy ",accuracy)


#main function initiating nearest neighbor computations
def k_nearest_neighbour(train_data,train_label,test_data,original_label,k):
	test_label=[]
	answer_list=[]
	for i in range(0,len(test_data)):
		
		temp_label=distance_cal(train_data,train_label,test_data[i])
		test_label.append(temp_label)
		
	answer_list=accuracy_cal(test_label,original_label)
	
	return answer_list
	

#method to calculate distances (Euclidean and Hamming) between rows
def distance_cal(train_data,train_label,test_data):
	
	
	distance_list=[]

	for i in range(0,len(train_data)):


		distance=np.linalg.norm(train_data[i]-test_data)

		distance_list.append([distance,i])

	y_predict=label_cal(distance_list,train_data,train_label)
	
	return y_predict
	

#method to predict labels for test data based on the majority labels of it k nearest neighbors
def label_cal(distance_list,train_data,train_label):
	
	sorted_distance=sorted(distance_list,key=lambda x:x[0])
	
	k_list=[]
	for i in range(0,k):
		k_list.append(sorted_distance[i])

	
	label_list=[]
	for i in range(0,k):
		label_list.append(train_label[k_list[i][1]])
		
	count=Counter(label_list)
	
	new_label=count.most_common()[0][0]
	return new_label


#method implementing k-fold cross validation	
def k_fold(datamatrix,fold_size):
	temp_accuracy=0
	
	precision=0
	recall=0
	F=0
	np.random.shuffle(datamatrix)
	answer_list=[]

	for i in range(0,k_folds):
		
		test_data=datamatrix[i*fold_size:i*fold_size+fold_size,:]
		train_data=np.delete(datamatrix,np.s_[i*fold_size:i*fold_size+fold_size],0)
		X_test=test_data[:,:col-1] 
		test_label=test_data[:,col-1]
		X_train=train_data[:,:col-1] 
		y_train=train_data[:,col-1]
		std_scale = preprocessing.StandardScaler().fit(X_train)
		X_train = std_scale.transform(X_train)
		X_test = std_scale.transform(X_test)
		answer_list=k_nearest_neighbour(X_train,y_train,X_test,test_label,k)
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




