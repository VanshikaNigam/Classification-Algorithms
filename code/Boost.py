import numpy as np
import pandas as pd
import sys 
import math
import random
import simplejson
import json
discrete_col =list()
def handle_categorical_data(file):
	columns=file.columns.values

	for column in columns:
		text_digit_vals={}
		def convert_to_int(val):
			return text_digit_vals[val]

		if file[column].dtype!=np.int64 and file[column].dtype!=np.float64:
			discrete_col.append(column)
			column_contents=file[column].values.tolist()
			uniques=set(column_contents)
			x=0
			for unique in uniques:
				if unique not in text_digit_vals:
					text_digit_vals[unique]=x
					x+=1
			file[column]=list(map(convert_to_int,file[column]))

	return file
file_name= 'project3_dataset2.txt'
file=pd.read_csv(file_name,sep='\t',header=None)
file=handle_categorical_data(file)
Gini=0
str_set=set()
data_matrix=file.as_matrix()
row,col=data_matrix.shape


###### K- Fold Cross Validation #########
k_folds=10
fold_size=int(row/10.0)
no_of_trees=5
###### K- Fold Cross Validation #########

class TreeNode(object):
  
  def __init__(self, gini, split_a,max_split_val):
  	self.gini=gini  	
  	self.attribute=split_a;
  	self.val = max_split_val
  	self.left_s = None
  	self.right_s = None
  	self.label=None

class decision_Tree(object):


	def makingtree(self,selected_data,test_matrix,nfolddata):

		print("Training the data:")
		root=self.build_a_tree(selected_data,1)

		predictions=self.error_predict(root,nfolddata,test_matrix) #tree,alpha,nfoldupdatedweights returned
		original_label_list=list()
		for i in range(0,test_matrix.shape[0]):

			original_label_list.append(test_matrix[i][-1])
		
		print("Predicting the labels:")
		label_for_test=self.predict(root,test_matrix,original_label_list)
		return label_for_test,predictions[1],predictions[2]


	def accuracy_cal(self,test_label,original_label):
		x=0
		true_positive=0
		true_negative=0
		false_positive=0
		false_negative=0
		for i in range(0,len(original_label)):
			if original_label[i]==1 and test_label[i]==1:
				true_positive=true_positive+1
			if original_label[i]==0 and test_label[i]==0:
				true_negative=true_negative+1
			if original_label[i]==0 and test_label[i]==1:
				false_positive=false_positive+1
			if original_label[i]==1 and test_label[i]==0:
				false_negative=false_negative+1

		if true_positive==0 and true_negative==0 and false_negative==0 and false_positive==0:
			accuracy=0
		else:
			accuracy=(true_negative+true_positive)/(true_positive+true_negative+false_negative+false_positive)
		
		if true_positive + false_positive > 0:
			precision = true_positive/(true_positive+false_positive)
		else :
			precision=0

		if true_positive + false_negative > 0 :
			recall=true_positive/(true_positive+false_negative)
			
		else:
			recall = 0

		if precision!=0 or recall!=0 : 
			F_Measure=2*precision*recall/(precision+recall)
		else:
			F_Measure = 0
		
		print("accuracy inside accuracy cal ",accuracy)
		print("precision inside accuracy cal ",precision)
		print("recall inside accuracy cal ",recall)
		print("f-measure inside accuracy cal ",F_Measure)
		
		return accuracy,precision,recall,F_Measure		

	def entire_Gini(self,data_matrix):
		r,c=data_matrix.shape
		label_as_One=0
		label_as_Zero=0
		for i in range(0,r):
			if data_matrix[i][-2]==1:
				label_as_One=label_as_One+1
			else:
				label_as_Zero=label_as_Zero+1
		Gini=1-((label_as_Zero/r)**2+(label_as_One/r)**2)
		
		return Gini

	def gini(self,data_matrix,attribute,val,gini_t):

		r,c=data_matrix.shape
		index_col= data_matrix[:,attribute]
		class_label=data_matrix[:,-2]
		less_than_val_count=0;
		more_than_val_count=0;
		p_less_group=0;
		p_more_group=0;
		less_than_list=list()
		more_than_list=list()

		for i in range (0, r):
			if index_col[i]<val:

				less_than_val_count=less_than_val_count+1
				less_than_list.append(data_matrix[i])

			else:
				more_than_val_count=more_than_val_count+1
				more_than_list.append(data_matrix[i])

		for i in range(0, r):
			if class_label[i]==1 and index_col[i]<val:
				p_less_group=p_less_group+1;
			if class_label[i]==1 and index_col[i]>=val:
				p_more_group=p_more_group+1;

		try:
			p_less_ratio= p_less_group/less_than_val_count
		except ZeroDivisionError:
			p_less_ratio = 0.0

		q_less_ratio=1-p_less_ratio

		try:
			p_more_ratio=p_more_group/more_than_val_count
		except ZeroDivisionError:
			p_more_ratio = 0.0
		q_more_ratio= 1-p_more_ratio

		gini_gain=gini_t-(((less_than_val_count/float(r))*(1-(p_less_ratio**2)- (q_less_ratio**2))) + ((more_than_val_count/float(r))*(1-(p_more_ratio**2)-(q_more_ratio**2))))
		
		return [gini_gain,less_than_list,more_than_list]

	def split(self,data_matrix):
		
		gini_t=self.entire_Gini(data_matrix)
		#print("EntireGini:",gini_t)
		split_attribute=0
		left_split=list()
		right_split=list()
		r,c= data_matrix.shape
		max_gini=-sys.maxsize - 1
		max_split_val=0


		list_range=list(range(0,data_matrix.shape[1]-2))
		
		no_of_cols_to_select=math.ceil((c-2)*0.2)
		
		idx_col=random.sample(list_range,no_of_cols_to_select)
		
		
		for index in idx_col:
			
			if index in discrete_col:
				if index in str_set:
					#print("inside")
					list_range.remove(index)
					idx_col=random.sample(list_range,no_of_cols_to_select)
				else:
					str_set.add(index)
		
		for attribute in idx_col:
			
			sorted_matrix=sorted(data_matrix,key=lambda x:x[attribute])
			sorted_matrix=np.asarray(sorted_matrix)
			att_from_sorted=sorted_matrix[:,attribute]

			max_gini_value=-sys.maxsize-1  
			cont_split_val=0
			for i in att_from_sorted: #np.unique
				gini_list=self.gini(data_matrix,attribute,i,gini_t)
				gini_list=np.array(gini_list,dtype=object)
		
				gini_gain=gini_list[0]
				l_split_cont=gini_list[1]
				r_split_cont= gini_list[2]
				cont_split_val=i
	
				if(float(gini_gain)>max_gini_value):
					max_gini_value=float(gini_gain)
					group1=l_split_cont
					group2=r_split_cont
					split_val=cont_split_val
			
			if max_gini_value>max_gini:
				max_gini=max_gini_value
				left_split=group1
				right_split=group2
				split_attribute=attribute
				max_split_val=split_val

		#print("Final max",max_gini)	
		#print("index of splitting attribute", split_attribute)
		#print("splitting val" , max_split_val)		

		return max_gini,split_attribute,max_split_val,left_split,right_split

	
	
	def make_terminal(self,data):
		
		labels = np.array([rw[-2] for rw in data])
		labels =labels.astype(int)
		
		unique, counts = np.unique(labels, return_counts=True)
		count = zip(unique, counts)
		count = sorted(count, key = lambda x: x[1], reverse=True)
		max_value = count[0][0]
		
	
		return max_value


	def build_a_tree(self,data_matrix,depth):
	
		max_gini, split_att, max_split_val,left,right=self.split(data_matrix)

		left=np.asarray(left)
		right=np.asarray(right)
		root = TreeNode(max_gini,split_att,max_split_val)
		
		left_len= len(left)
		right_len=len(right)

		
		if left_len==0 or right_len==0:
			if left_len==0:
				root.label=self.make_terminal(right)
				return root

			elif right_len==0:
				root.label=self.make_terminal(left)
				return root
	
		root.left_s = self.build_a_tree(left,depth+1)
		root.right_s = self.build_a_tree(right, depth+1)
	
		return root

	def predict_classification(self,node,row):
		current=node
	
		if row[node.attribute]<node.val:
			
		
			node=node.left_s
			if node!=None:
				predicted_label=self.predict_classification(node,row)
				
				return predicted_label
				
			else:
				
				return current.label
		else :
			
			node=node.right_s
			
			if node!=None:
				
				predicted_label=self.predict_classification(node,row)
				
				return predicted_label
				
			else:
				
				return current.label
	
	def error_predict(self,root,nfolddata,test_matrix):
		o_list=list()
		
		for i in range(0,nfolddata.shape[0]):
			o_list.append(int(nfolddata[i][-2]))
		
		p_label_list=list()
		for eachrow in nfolddata:

			predict_label=self.predict_classification(root,eachrow)
			
			p_label_list.append(predict_label)
		
		misclassification_list=list()
		weighted_matrix_Sum=np.sum(nfolddata[:,-1]) #find the combined weight of the last column
		
		for i in range(0,len(o_list)):
			if o_list[i]!=p_label_list[i]:
				misclassification_list.append(nfolddata[i][-1])

		error=sum(misclassification_list)/weighted_matrix_Sum	
		alpha=0.5*math.log( float(1-error) / float(error))	
		#print("alpha:",alpha)

		for i in range(0,len(o_list)):
			
			if o_list[i]==p_label_list[i]:
				
				nfolddata[i][-1]=nfolddata[i][-1]*math.exp(-alpha*1)/weighted_matrix_Sum
			else:
				
				nfolddata[i][-1]=nfolddata[i][-1]*math.exp(-alpha*-1)/weighted_matrix_Sum 
		
		return root,alpha,nfolddata
		

	def predict(self,node,test_matrix,original_label_list):
		Accuracy=0.0
		Accuracy_list=list()
		predicted_label_list=list()
		for row in test_matrix:

			predicted_label=self.predict_classification(node,row)
		
		
			predicted_label_list.append(predicted_label)
		
		return predicted_label_list

	def boosting(self,data,test_matrix):
		
		num_rows,num_cols=data.shape
		original_label_list=list()
		r_test,c_test=test_matrix.shape
		
		for i in range(0,r_test):
			original_label_list.append(test_matrix[i][-1])

		initial_weights=(1.0/num_rows)
		weighted_array=np.full(num_rows ,initial_weights)
		weighted_array=weighted_array.reshape((num_rows,1))
		data=np.hstack((data,weighted_array))

		label_mat=np.zeros((test_matrix.shape[0],1))
		alpha_list=list()
		label_array=list()
		for i in range(0,no_of_trees):
			data_percent=int(data.shape[0]*0.632)
			idx= random.choices(range(0,data.shape[0]), weights=data[:,-1], k=data_percent)
			data_part1= data[idx,:]
			data_remaining_percent= data.shape[0] - len(idx)
			idx2=random.sample(range(0,len(idx)), data_remaining_percent)
			data_part2=data_part1[idx2,:]
			total_data= np.vstack((data_part1,data_part2))
			

			result=self.makingtree(total_data,test_matrix,data) # for testing the tree made
			
			data=result[2]
			alpha_list.append(result[1])
			labels=np.asarray(result[0])
			labels=labels.reshape(len(result[0]),1)
			label_mat= np.concatenate((label_mat, labels), axis=1)


		label_mat=np.delete(label_mat, 0, 1)
		label_for_1_wts=0
		label_for_0_wts=0
		
		for row in range(0,label_mat.shape[0]):
			for column in range(0,label_mat.shape[1]):
				if label_mat[row][column]==1:
					#print("col:",column)
					label_for_1_wts=label_for_1_wts+alpha_list[column]
				else:
					#print("col:",column)
					label_for_0_wts=label_for_0_wts+alpha_list[column]

			if label_for_1_wts>label_for_0_wts:
				label_array.append(1)
			else:
				label_array.append(0)

		accuracy_val=self.accuracy_cal(label_array,original_label_list)

		return accuracy_val

	
	def k_fold(self,data_matrix,fold_size):
	
		Accuracy_list=list()
		Precision_list=list()
		Recall_list=list()
		F_Measure_list=list()

		for i in range(0,k_folds):
			test_label=list()
			testdata_matrix=data_matrix[i*fold_size:i*fold_size+fold_size,:]
			traindata_matrix=np.delete(data_matrix,np.s_[i*fold_size:i*fold_size+fold_size],0)
			result=self.boosting(traindata_matrix,testdata_matrix)
			Accuracy_list.append(result[0])
			Precision_list.append(result[1])
			Recall_list.append(result[2])
			F_Measure_list.append(result[3])

		print("*********Final Results*************")
		avg_accuracy=(sum(Accuracy_list)/10.0)
		avg_precision=(sum(Precision_list)/10.0)
		avg_recall=(sum(Recall_list)/10.0)
		avg_fmeasure=(sum(F_Measure_list)/10.0)
		print("Average accuracy is: ", avg_accuracy)
		print("Average precision is: ", avg_precision)
		print("Average recall is: ", avg_recall)
		print("Average f-measure is: ", avg_fmeasure)


obj= decision_Tree()
obj.k_fold(data_matrix,fold_size)
