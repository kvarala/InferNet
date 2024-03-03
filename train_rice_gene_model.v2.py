from prep_data import prep_data, asign_classes_to_probabilities
from prepare_data import prepare_data
from calculate_metrics import calculate_metrics
from joblib import load, dump
import pickle
import sys
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import linear_model
from sklearn.metrics import accuracy_score, confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

# We use OneVsRestClassifier for multi-label prediction
from sklearn.multiclass import OneVsRestClassifier
orderedClasses = ['anther', 'leaf', 'root', 'seed', 'seedling', 'shoot']
rice_mapping, X, Y, rice_req_sra_ids, final_gene_names= prep_data("training_samples.txt", "Oryza.Norm.Log2.txt")
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y)

# Train an SVM based classifier on the rice dataset
svm_classifier = SVC(random_state=42, kernel='poly', probability=True)
print ("Beginning training SVM")
svm_classifier.fit(X_train, Y_train)
print ("SVM Training complete")
dump(svm_classifier, './all_genes_rice_proba_svm_model.sav')

# Reencode the categorical labels as numeric labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_validation = le.fit_transform(Y_validation)

# Train an xgboost classifier, max_depth is 3 and objective is multi:softprob. All other parameters are default for sklearn's XGBClassifier
xgb_classifier = XGBClassifier(max_depth=3, objective='multi:softprob')
print ("Beginning training XGB")
xgb_classifier.fit(X_train, Y_train,verbose=1,eval_set=[(X_train, Y_train), (X_validation, Y_validation)])
print ("XGB training complete")
dump(xgb_classifier, './all_genes_rice_proba_xgb_model.sav')

y_pred = xgb_classifier.predict(X_validation)


# Print the confusion Matrix and classification report'
print(confusion_matrix(Y_validation, y_pred))
print(classification_report(Y_validation, y_pred))
fig, ax = plt.subplots(figsize=(10, 5))
ConfMat=ConfusionMatrixDisplay.from_predictions(Y_validation, y_pred, ax=ax,labels=xgb_classifier.classes_,display_labels=orderedClasses)
ConfMat.plot(cmap=plt.cm.Blues)
ConfMat.figure_.savefig('xgb_ConfusionMatrix.pdf')

