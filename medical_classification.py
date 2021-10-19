import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statistics import mean, mode
import pandas as pd
import medical_lib as ml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost.core import Booster
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#################################################################################################
medData = pd.read_csv('dfNumbersOnly.csv')
medDataCopy = medData.copy()

names = medDataCopy.iloc[:, 2]
medDataCopy = medDataCopy.iloc[:, 3:]
medDataCopy = medDataCopy.fillna(1)
#################################################################################################
med_class = medDataCopy.iloc[:, -1]

med_features = medDataCopy.iloc[:, :-1]

###########################################################################################################################
###########################################################################################################################
# Aufteilen der Daten in 4 Untersets
med_features_train, med_features_test, med_class_train, med_class_test = train_test_split(
    med_features, med_class, test_size=0.2, random_state=43, stratify=med_class)
med_class_test_array = np.array(med_class_test)
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# XGBoost
# xgmodel = XGBClassifier(eval_metric='error')
# xgmodel.fit(med_features_train, med_class_train)
# #print(xgmodel) zeigt die Parameter des Classifiers an
# xgboosted_prediction = xgmodel.predict(med_features_test)
# acc_CV = cross_val_score(xgmodel, med_features, med_class, cv=10)
# print(acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
# # print(xgmodel.feature_importances_)
# plot_importance(xgmodel)
# pyplot.show()
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

# # Knn-Classifier for K up to 200
medKNN = KNeighborsClassifier(n_neighbors=8)
#Training
medKNN.fit(med_features_train,med_class_train)
knnYpred = medKNN.predict(med_features_test)
accuracyKNN, precisionKNN, recallKNN, f1scoreKNN = ml.scoring(knnYpred, med_class_test_array)
print('KNN Accuracy: ', accuracyKNN, 'KNN Precision: ', precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN )
print('')
###########################################################################################################################
###########################################################################################################################
# Jetzt mit n-fold Cross-Validation
medical_KFOLD_KNN = KNeighborsClassifier(n_neighbors=8)
# Training mit n-foldaccuracyKNN_CV, precisionKNN_CV, recallKNN_CV, f1scoreKNN_CV
accuracyKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10)
precisionKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='precision')
recallKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='recall')
f1scoreKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='f1')
meanAccuracyKNN_CV = np.mean(accuracyKNN_CV)
meanPrecisionKNN_CV = np.mean(precisionKNN_CV)
meanRecallKNN_CV = np.mean(recallKNN_CV)
meanF1scoreKNN_CV = np.mean(f1scoreKNN_CV)
print('10-Fold KNN Accuracy: ', meanAccuracyKNN_CV, '10-Fold KNN Precision: ', meanPrecisionKNN_CV, '10-Fold KNN Recall: ', meanRecallKNN_CV, '10-Fold KNN F1-Score: ', meanF1scoreKNN_CV )
print('')
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# Logistic Regression:
lr_model = LogisticRegression()
lr_model.fit(med_features_train, med_class_train)
#Prediction of test set
lr_y_pred = lr_model.predict(med_features_test)
lr_accuracyLogReg, lr_precisionLogReg, lr_recallLogReg, lr_f1scoreLogReg = ml.scoring(lr_y_pred, med_class_test_array)
print('Log-Regression Accuracy: ', lr_accuracyLogReg, 'Log-Regression Precision: ', lr_precisionLogReg, 'Log-Regression Recall: ', lr_recallLogReg, 'Log-Regression F1-Score: ', lr_f1scoreLogReg )
print('')


# 10-Fold Logistic Regression:
medical_KFOLD_LogReg = LogisticRegression()
accuracyLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10)
precisionLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='precision')
recallLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='recall')
f1scoreLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='f1')
meanAccuracyLogReg_CV = np.mean(accuracyLogReg_CV)
meanPrecisionLogReg_CV = np.mean(precisionLogReg_CV)
meanRecallLogReg_CV = np.mean(recallLogReg_CV)
meanF1scoreLogReg_CV = np.mean(f1scoreLogReg_CV)
print('10-Fold LogReg Accuracy: ', meanAccuracyLogReg_CV, '10-Fold LogReg Precision: ', meanPrecisionLogReg_CV, '10-Fold LogReg Recall: ', meanRecallLogReg_CV, '10-Fold LogReg F1-Score: ', meanF1scoreLogReg_CV )
print('')

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# # SVM:
# # Create a svm Classifier
medical_SVM = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
medical_SVM.fit(med_features_train, med_class_train)
#Predict the response for test dataset
svmPred = medical_SVM.predict(med_features_test)
accuracySVM, precisionSVM, recallSVM, f1scoreSVM = ml.scoring(svmPred, med_class_test_array)
print('SVM Accuracy: ', accuracySVM, 'SVM Precision: ', precisionSVM, 'SVM Recall: ', recallSVM, 'SVM F1-Score: ', f1scoreSVM )
print('')


#10-Fold SVM
medical_KFOLD_SVM = svm.SVC(kernel='linear')
accuracySVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10)
precisionSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='precision')
recallSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='recall')
f1scoreSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='f1')
meanAccuracySVM_CV = np.mean(accuracySVM_CV)
meanPrecisionSVM_CV = np.mean(precisionSVM_CV)
meanRecallSVM_CV = np.mean(recallSVM_CV)
meanF1scoreSVM_CV = np.mean(f1scoreSVM_CV)
print('10-Fold SVM Accuracy: ', meanAccuracySVM_CV, '10-Fold SVM Precision: ', meanPrecisionSVM_CV, '10-Fold SVM Recall: ', meanRecallSVM_CV, '10-Fold SVM F1-Score: ', meanF1scoreSVM_CV )
print('')

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#Decision Tree
