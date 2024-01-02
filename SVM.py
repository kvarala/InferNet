#author: Somaiah Thimmaiah Balekuttira
#Location:Varala Labs, Purdue University
#Date:July 11, 2018
#Day:Friday
#Email:sbalekut@purdue.edu


import numpy as np
import sklearn                         #Builtin KNN
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
import pickle
import sys
from sklearn.svm import SVC


#Pre processes the data from the text file as well as connects the labels to samples
def pre_processing():
    
	lines=[]
	with open('Training_data.txt',"r") as f:			#Has the gene values for each experiment
		for line in f:
	    		lines.append(line.strip().split("\t"))

 
	print len(lines[0])
	print len(lines[1])
	sample_ids=[]
	#labels=[]
	for sample in range(len(lines[0])):
	    sample_ids.append(lines[0][sample])

	print len(sample_ids)

	sample_values=[]
	for i in range(len(lines[0])+1):									#Plus one because the geneid adds one to length 
		temp=[]
		for j in range(len(lines)):
		    if(i!=0 and j!=0):
			temp.append(lines[j][i])
		sample_values.append(temp)

	sample_values = [x for x in sample_values if x != []]
	print len(sample_values)
	
	if(len(sample_ids) != len(sample_values)):
        	sys.exit("SOME SAMPLE VALUES NOT FOUND")
	
	labels_expe=[]						#Labels and exp order not maintained
	with open('training_samples.txt',"r") as l:			#Has the labels for each experiment
		for line in l:
	    		labels_expe.append(line.strip().split("\t")) 

	
	labels_ordered=[]

	for exp in sample_ids:
		flag=0
		for i in range(len(labels_expe)):
			if(exp==labels_expe[i][0]):
				flag=1
				labels_ordered.append(labels_expe[i][1])
		if(flag==0):
			print exp
			sys.exit("SOME SAMPLE LABEL NOT FOUND")

#	print sample_values,
#	print labels_ordered
#	print sample_ids	
	
	return sample_values,labels_ordered,sample_ids					


	 



#Splits the data into train and test
def split(sample_values,labels):
    X=np.asarray(sample_values)
    y=np.asarray(labels)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1)
    print "Training samples:",
    print len(X_train) 
    print "Testing samples:",
    print len(X_test)
    return X_train, X_test, y_train, y_test





#Trains the classifier on test data
def train_test(X_train, X_test, y_train, y_test):
    classifier=SVC(probability=True)
    classifier.fit(X_train,y_train)

    accuracy=classifier.score(X_test,y_test)
    print "Accuracy:",
    print accuracy
    return classifier



'''
#Predicts the label of a unseen sample using the trained classifier
def predict_sample(classifier,sample_test):
    example_measures=sample_test	 #reshape(1,len(sample_test))
    return classifier.predict(example_measures), classifier.predict_proba(example_measures)


def make_test_sample():
	lines=[]
	with open('processed_data_slq.txt',"r") as f:
	      	for line in f:
           		 lines.append(line.strip().split("\t")) 
	
	for samples in range(len(lines[0])):
		a = np.zeros(shape=(1,len(lines)-1))
		count=0
		
		for i in range(len(lines)):
			if(i!=0):
				a[0][count]=lines[i][samples+1]
				count=count+1						
		prediction,proba=predict_sample(trained_classifier,a)
		print prediction,
		print proba,
		print lines[0][samples]
'''	
#Code entry point
sample_values,labels,sample_ids=pre_processing()
X_train, X_test, y_train, y_test=split(sample_values,labels)
trained_classifier=train_test(X_train, X_test, y_train, y_test)
pickle.dump(trained_classifier, open('svm_rnaseq_model_saved.sav', 'wb'))
#make_test_sample()





