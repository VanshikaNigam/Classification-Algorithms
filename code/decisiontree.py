import numpy as np
import pandas as pd
import sys 
import math
import random

#### Handling Categorical Data #####

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
	#print(file)

	return file

file_name= 'project3_dataset2.txt'
file=pd.read_csv(file_name,sep='\t',header=None)
file=handle_categorical_data(file)
Gini=0
data_matrix=file.as_matrix()
row,col=data_matrix.shape

###### K- Fold Cross Validation #########
k_folds=10
fold_size=int(row/10.0)
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

	def main(self,data_matrix,test_matrix):
		print("Training of the data ")
		root= self.build_a_tree(data_matrix,1)
		original_label_list=list()
		r,c=test_matrix.shape
		for i in range(0,r):

			original_label_list.append(test_matrix[i][-1])
		#print("original_label_list:",original_label_list)
		print("Predicting the lables")
		accuracy_result=self.predict(root,test_matrix,original_label_list)
		return accuracy_result


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
			F_Measure=(2*precision*recall)/(precision+recall)
			
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
			if data_matrix[i][-1]==1:
				label_as_One=label_as_One+1
			else:
				label_as_Zero=label_as_Zero+1
		Gini=1-((label_as_Zero/r)**2+(label_as_One/r)**2)
		
	#print("gini ", Gini)
		return Gini

	def gini(self,data_matrix,attribute,val,gini_t):

		r,c=data_matrix.shape
		class_label = data_matrix[:,-1]

		index_col= data_matrix[:,attribute]
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
	
		attributes = np.random.choice(range(c-1),c-1,replace=False)
		attributes.sort()
		
		for attribute in attributes:
			
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

		"""print("Final max",max_gini)	
		print("index of splitting attribute", split_attribute)
		print("splitting val" , max_split_val)"""		

		return max_gini,split_attribute,max_split_val,left_split,right_split

	
	def make_terminal(self,data):
		#print("Inside make_terminal()")
		labels = np.array([rw[-1] for rw in data])
		labels =labels.astype(int)
		#print("inside terminal1:",labels)
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
		"""print("Left Split array")
		print(left)
		print(left.shape)
		print("Right Split array")
		print(right.shape)
		print(right)"""
		
		left_len= len(left)
		right_len=len(right)
		#print("length:",left_len,right_len)

		### Post Processing Steps ###

		"""if(depth>=max_depth):
			#print("inside depth=max_depth")
			#print("dimension of left", left.shape)
			#print("dimension of right", right.shape)
			if left_len!=0 and right_len!=0:
				terminal_data= np.vstack((left, right))
				root.label=self.make_terminal(terminal_data)
				#print("inside max depth if condition:",root.val)
				return root
			elif left_len==0:
				root.label=self.make_terminal(right)
				#print("inside max depth elif 1 condition:",root.val)
				return root
			elif right_len==0:
				root.label=self.make_terminal(left)
				#print("inside max depth elif 2 condition:",root.val)"""

				### Post Processing Steps ###


		if left_len==0 or right_len==0:
			if left_len==0:
				#print("insdie left_len=0:",right)
				root.label=self.make_terminal(right)
				#print("inside max depth elif 3 condition:",root.val)
				return root

			elif right_len==0:
				#print("insdie right_len=0:",left)
				root.label=self.make_terminal(left)
				return root
		
				### Post Processing Steps ###

		"""data_on_root= np.vstack((left, right))
		if len(data_on_root)<=min_data_on_node:
				#print("INSIDE MINIMUM DATA CONDITION")
				
				root.label=self.make_terminal(data_on_root)	
				return root	"""		

				### Post Processing Steps ###
	
		root.left_s = self.build_a_tree(left,depth+1)
		root.right_s = self.build_a_tree(right, depth+1)
	
		return root

	def predict_classification(self,node, row):
		#print("node.attribute",node.attribute)
		#print("row[node.attribute] :", [node.attribute])
	
		current=node
	
		if row[node.attribute]<node.val:
			#print("node,left_s:",node.val)
		
			node=node.left_s
			if node!=None:
				predicted_label=self.predict_classification(node,row)
				#print("predicted_label not none:",predicted_label)
				return predicted_label
				#print("predicted_label:",predicted_label)
			else:
				#print("current label none:",current.label)
				return current.label
		else :
			#print("node,right_s:",node.val)
			node=node.right_s
			
			if node!=None:
				#("Inside node!=None")
				predicted_label=self.predict_classification(node,row)
				#print("predicted_label right not none:",predicted_label)
				return predicted_label
				#print("predicted_label:",predicted_label)
			else:
				return current.label
			

	def predict(self,node,test_matrix,original_label_list):
		Accuracy=0.0
		Accuracy_list=list()
		predicted_label_list=list()
		for row in test_matrix:

			predicted_label=self.predict_classification(node,row)
			predicted_label_list.append(predicted_label)
		
		accuracy_val=self.accuracy_cal(predicted_label_list,original_label_list)
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
			result=self.main(traindata_matrix,testdata_matrix)
			Accuracy_list.append(result[0])
			Precision_list.append(result[1])
			Recall_list.append(result[2])
			F_Measure_list.append(result[3])
		
		print("*********Final Results*************")
		avg_accuracy=(sum(Accuracy_list)/10.0)
		avg_precision=(sum(Precision_list)/10.0)
		avg_recall=(sum(Recall_list)/10.0)
		avg_fmeasure=(sum(F_Measure_list)/10.0)
		print("Average Accuracy is: ", avg_accuracy)
		print("Average Precision is: ", avg_precision)
		print("Average Recall is: ", avg_recall)
		print("Average F-Measure is: ", avg_fmeasure)


obj= decision_Tree()
obj.k_fold(data_matrix,fold_size)









