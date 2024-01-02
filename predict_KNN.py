import xlrd                            #For excel sheet processing
import numpy as np
import sklearn                         #Builtin KNN
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt



#Predicts the label of a unseen sample using the trained classifier
def predict_sample(classifier,sample_test):
    example_measures=sample_test	 #reshape(1,len(sample_test))
    return classifier.predict(example_measures), classifier.predict_proba(example_measures)


def make_test_sample(filename,model):


	#File to write the output
	out=open("rnaseq_KNN_results.txt",'w')


	lines=[]
	with open(filename,"r") as f:
	      	for line in f:
           		 lines.append(line.strip().split("\t")) 
	
	for samples in range(len(lines[0])):
		a = np.zeros(shape=(1,len(lines)-1))
		count=0
		
		for i in range(len(lines)):
			if(i!=0):
				a[0][count]=lines[i][samples+1]
				count=count+1						
		prediction,proba=predict_sample(model,a)	
		out.write(str(prediction))
		out.write("\t")
		out.write(str(proba))
		out.write("\t")
		out.write(lines[0][samples])
		out.write("\n")



saved_model='knn_rnaseq_model_saved.sav'
loaded_model = pickle.load(open(saved_model, 'rb'))
filename='Testing_data.txt'
make_test_sample(filename,loaded_model)
