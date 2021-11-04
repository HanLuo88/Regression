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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
#################################################################################################
# medData = pd.read_csv('naive_latest.csv')
# medDataCopy = medData.copy()
# medDataCopy = medDataCopy.iloc[:, 3:]



# #################################################################################################
# med_class = medDataCopy.iloc[:, -1]

# med_features = medDataCopy.iloc[:, :-1]

# ###########################################################################################################################
# ###########################################################################################################################
# # Aufteilen der Daten in 4 Untersets
# med_features_train, med_features_test, med_class_train, med_class_test = train_test_split(
#     med_features, med_class, test_size=0.2, random_state=43, stratify=med_class)
# med_class_test_array = np.array(med_class_test)
###########################################################################################################################
###########################################################################################################################
medData = pd.read_csv('naive_latest_selection.csv')
medDataCopy = medData.copy()
medDataCopy = medDataCopy.iloc[:, 3:]
print(medDataCopy.head())
medDataCopy_model2_Features_Selected = medDataCopy.copy()

#################################################################################################
med_class = medDataCopy.iloc[:, -1]

med_features = medDataCopy.iloc[:, :-1]

###########################################################################################################################
###########################################################################################################################
# Aufteilen der Daten in 4 Untersets
med_features_train, med_features_test, med_class_train, med_class_test = train_test_split(med_features, med_class, test_size=0.2, random_state=43, stratify=med_class)
med_class_test_array = np.array(med_class_test)

p1235_M1 = pd.read_csv('p1235_M1_selection.csv')
p1235_M1 = p1235_M1.iloc[:, 3:]
print(p1235_M1.columns)
p3487_M1 = pd.read_csv('p3487_M1_selection.csv')
p3487_M1 = p3487_M1.iloc[:, 3:]
p5865_M1 = pd.read_csv('p5865_M1_selection.csv')
p5865_M1 = p5865_M1.iloc[:, 3:]
p8730_M1 = pd.read_csv('p8730_M1_selection.csv')
p8730_M1 = p8730_M1.iloc[:, 3:]
###########################################################################################################################
###########################################################################################################################
#Ohne Boost
print('Nachfolgend sind alle Vorhersagen ohne Featureboost')
# # Knn-Classifier for K = 8
print('')
medKNN = KNeighborsClassifier(n_neighbors=8)
#Training
medKNN.fit(med_features_train,med_class_train)

medKNN_pred1 = medKNN.predict(p1235_M1)
print('KNN: ', medKNN_pred1)

medKNN_pred2 = medKNN.predict(p3487_M1)
print('KNN: ', medKNN_pred2)

medKNN_pred3 = medKNN.predict(p5865_M1)
print('KNN: ', medKNN_pred3)

medKNN_pred4 = medKNN.predict(p8730_M1)
print('KNN: ', medKNN_pred4)

knnYpred = medKNN.predict(med_features_test)
accuracyKNN, precisionKNN, recallKNN, f1scoreKNN = ml.scoring(knnYpred, med_class_test_array)
print('KNN Accuracy: ', accuracyKNN, 'KNN Precision: ', precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN )
print('#################################################################################################')
# ###########################################################################################################################
# ###########################################################################################################################
# # Jetzt mit n-fold Cross-Validation
print('10-Fold KNN')
print('')
medical_KFOLD_KNN = KNeighborsClassifier(n_neighbors=8)
# Training mit n-foldaccuracyKNN_CV, precisionKNN_CV, recallKNN_CV, f1scoreKNN_CV
accuracyKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='accuracy')
precisionKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='precision')
recallKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='recall')
f1scoreKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features, med_class, cv=10, scoring='f1')
meanAccuracyKNN_CV = np.mean(accuracyKNN_CV)
meanPrecisionKNN_CV = np.mean(precisionKNN_CV)
meanRecallKNN_CV = np.mean(recallKNN_CV)
meanF1scoreKNN_CV = np.mean(f1scoreKNN_CV)
print('10-Fold KNN Accuracy: ', meanAccuracyKNN_CV, '10-Fold KNN Precision: ', meanPrecisionKNN_CV, '10-Fold KNN Recall: ', meanRecallKNN_CV, '10-Fold KNN F1-Score: ', meanF1scoreKNN_CV )
print('#################################################################################################')
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# # Logistic Regression:
print('Logistic Regression')
print('')
lr_model = LogisticRegression()
lr_model.fit(med_features_train, med_class_train)

lr_model_pred1 = lr_model.predict(p1235_M1)
print('Logistic Regression: ', lr_model_pred1)

lr_model_pred2 = lr_model.predict(p3487_M1)
print('Logistic Regression: ', lr_model_pred2)

lr_model_pred3 = lr_model.predict(p5865_M1)
print('Logistic Regression: ', lr_model_pred3)

lr_model_pred4 = lr_model.predict(p8730_M1)
print('Logistic Regression: ', lr_model_pred4)

#Prediction of test set
lr_y_pred = lr_model.predict(med_features_test)
lr_accuracyLogReg, lr_precisionLogReg, lr_recallLogReg, lr_f1scoreLogReg = ml.scoring(lr_y_pred, med_class_test_array)
print('Log-Regression Accuracy: ', lr_accuracyLogReg, 'Log-Regression Precision: ', lr_precisionLogReg, 'Log-Regression Recall: ', lr_recallLogReg, 'Log-Regression F1-Score: ', lr_f1scoreLogReg )
print('#################################################################################################')


# # 10-Fold Logistic Regression:
print('10-Fold Logistic Regression')
print('')
medical_KFOLD_LogReg = LogisticRegression()
accuracyLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='accuracy')
precisionLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='precision')
recallLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='recall')
f1scoreLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features, med_class, cv=10, scoring='f1')
meanAccuracyLogReg_CV = np.mean(accuracyLogReg_CV)
meanPrecisionLogReg_CV = np.mean(precisionLogReg_CV)
meanRecallLogReg_CV = np.mean(recallLogReg_CV)
meanF1scoreLogReg_CV = np.mean(f1scoreLogReg_CV)
print('10-Fold LogReg Accuracy: ', meanAccuracyLogReg_CV, '10-Fold LogReg Precision: ', meanPrecisionLogReg_CV, '10-Fold LogReg Recall: ', meanRecallLogReg_CV, '10-Fold LogReg F1-Score: ', meanF1scoreLogReg_CV )
print('#################################################################################################')

# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# # # SVM:
# print('SVM')
# print('')
# # # Create a svm Classifier
# medical_SVM = svm.SVC(kernel='linear') # Linear Kernel
# #Train the model using the training sets
# medical_SVM.fit(med_features_train, med_class_train)

# medical_SVM_pred1 = medical_SVM.predict(p1235_M1)
# print('SVM: ', medical_SVM_pred1)

# medical_SVM_pred2 = medical_SVM.predict(p3487_M1)
# print('SVM: ', medical_SVM_pred2)

# medical_SVM_pred3 = medical_SVM.predict(p5865_M1)
# print('SVM: ', medical_SVM_pred3)

# medical_SVM_pred4 = medical_SVM.predict(p8730_M1)
# print('SVM: ', medical_SVM_pred4)

# #Predict the response for test dataset
# svmPred = medical_SVM.predict(med_features_test)
# accuracySVM, precisionSVM, recallSVM, f1scoreSVM = ml.scoring(svmPred, med_class_test_array)
# print('SVM Accuracy: ', accuracySVM, 'SVM Precision: ', precisionSVM, 'SVM Recall: ', recallSVM, 'SVM F1-Score: ', f1scoreSVM )
# print('#################################################################################################')


# #10-Fold SVM
# print('10-Fold SVM')
# print('')
# medical_KFOLD_SVM = svm.SVC(kernel='linear')
# accuracySVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='accuracy')
# precisionSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='precision')
# recallSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='recall')
# f1scoreSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features, med_class, cv=10, scoring='f1')
# meanAccuracySVM_CV = np.mean(accuracySVM_CV)
# meanPrecisionSVM_CV = np.mean(precisionSVM_CV)
# meanRecallSVM_CV = np.mean(recallSVM_CV)
# meanF1scoreSVM_CV = np.mean(f1scoreSVM_CV)
# print('10-Fold SVM Accuracy: ', meanAccuracySVM_CV, '10-Fold SVM Precision: ', meanPrecisionSVM_CV, '10-Fold SVM Recall: ', meanRecallSVM_CV, '10-Fold SVM F1-Score: ', meanF1scoreSVM_CV )
# print('#################################################################################################')

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#Decision Tree
print('Decision Tree')
print('')
medical_DecTree = DecisionTreeClassifier()
medical_DecTree = medical_DecTree.fit(med_features_train, med_class_train)

medical_DecTree_pred1 = medical_DecTree.predict(p1235_M1)
print('Decision Tree: ', medical_DecTree_pred1)

medical_DecTree_pred2 = medical_DecTree.predict(p3487_M1)
print('Decision Tree: ', medical_DecTree_pred2)

medical_DecTree_pred3 = medical_DecTree.predict(p5865_M1)
print('Decision Tree: ', medical_DecTree_pred3)
medical_DecTree_pred4 = medical_DecTree.predict(p8730_M1)
print('Decision Tree: ', medical_DecTree_pred4)

decTree_pred = medical_DecTree.predict(med_features_test)
accuracyDecTree, precisionDecTree, recallDecTree, f1scoreDecTree = ml.scoring(decTree_pred, med_class_test_array)
print('medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ', precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree )
print('#################################################################################################')

#10-Fold Decision Tree
print('10-Fold Decision Tree')
print('')
medical_KFOLD_DecTree = DecisionTreeClassifier()
accuracyDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features, med_class, cv=10, scoring='accuracy')
precisionDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features, med_class, cv=10, scoring='precision')
recallDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features, med_class, cv=10, scoring='recall')
f1scoreDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features, med_class, cv=10, scoring='f1')
meanAccuracyDecTree_CV = np.mean(accuracyDecTree_CV)
meanPrecisionDecTree_CV = np.mean(precisionDecTree_CV)
meanRecallDecTree_CV = np.mean(recallDecTree_CV)
meanF1scoreDecTree_CV = np.mean(f1scoreDecTree_CV)
print('10-Fold DecTree Accuracy: ', meanAccuracyDecTree_CV, '10-Fold DecTree Precision: ', meanPrecisionDecTree_CV, '10-Fold DecTree Recall: ', meanRecallDecTree_CV, '10-Fold DecTree F1-Score: ', meanF1scoreDecTree_CV )
print('#################################################################################################')

# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# #Random Forest
print('Random Forest')
print('')
# for estimator in range(50, 501, 25):
medical_RF = RandomForestClassifier(n_estimators= 50)
medical_RF.fit(med_features_train, med_class_train)

medical_RF_pred1 = medical_RF.predict(p1235_M1)
print('Random Forest: ', medical_RF_pred1)

medical_RF_pred2 = medical_RF.predict(p3487_M1)
print('Random Forest: ', medical_RF_pred2)

medical_RF_pred3 = medical_RF.predict(p5865_M1)
print('Random Forest: ', medical_RF_pred3)

medical_RF_pred4 = medical_RF.predict(p8730_M1)
print('Random Forest: ', medical_RF_pred4)

rfPred = medical_RF.predict(med_features_test)
accuracyRF, precisionRF, recallRF, f1scoreRF = ml.scoring(rfPred, med_class_test_array)
print('Anzahl Estimator: 50 ', 'RF Accuracy: ', accuracyRF, 'RF Precision: ', precisionRF, 'RF Recall: ', recallRF, 'RF F1-Score: ', f1scoreRF )
print('#################################################################################################')

# 10-Fold Random Forest
print('10-Fold Random Forest')
print('')
medical_KFOLD_RF = RandomForestClassifier(n_estimators= 50)
accuracyRF_CV = cross_val_score(medical_KFOLD_RF, med_features, med_class, cv=10, scoring='accuracy')
precisionRF_CV = cross_val_score(medical_KFOLD_RF, med_features, med_class, cv=10, scoring='precision')
recallRF_CV = cross_val_score(medical_KFOLD_RF, med_features, med_class, cv=10, scoring='recall')
f1scoreRF_CV = cross_val_score(medical_KFOLD_RF, med_features, med_class, cv=10, scoring='f1')
meanAccuracyRF_CV = np.mean(accuracyRF_CV)
meanPrecisionRF_CV = np.mean(precisionRF_CV)
meanRecallRFCV = np.mean(recallRF_CV)
meanF1scoreRF_CV = np.mean(f1scoreRF_CV)
print('10-Fold RF Accuracy: ', meanAccuracyRF_CV, '10-Fold RF Precision: ', meanPrecisionRF_CV, '10-Fold RF Recall: ', meanRecallRFCV, '10-Fold RF F1-Score: ', meanF1scoreRF_CV )
print('#################################################################################################')

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
# # # # # AdaBoost zum Klassifizieren
print('ADABOOST: ')
print('')
adamodel = AdaBoostClassifier()
adamodel.fit(med_features_train, med_class_train)

adamodel_prediction1 = adamodel.predict(p1235_M1)
print('AdaBoost: ', adamodel_prediction1)


adamodel_prediction2 = adamodel.predict(p3487_M1)
print('AdaBoost: ', adamodel_prediction2)


adamodel_prediction3 = adamodel.predict(p5865_M1)
print('AdaBoost: ', adamodel_prediction3)


adamodel_prediction4 = adamodel.predict(p8730_M1)
print('AdaBoost: ', adamodel_prediction4)
print('')

# #print(adamodel) zeigt die Parameter des Classifiers an
adamodel_prediction = adamodel.predict(med_features_test)
print('AdaBoost: ','predicted: \n', adamodel_prediction)
print('AdaBoost: ','Actual: \n', med_class_test.to_numpy())

adamodel_accuracy = accuracy_score(med_class_test, adamodel_prediction)
adamodel_precision = precision_score(med_class_test, adamodel_prediction, average='weighted')
adamodel_recall = recall_score(med_class_test, adamodel_prediction, average='weighted')
adamodel_f1 = f1_score(med_class_test, adamodel_prediction, average='weighted')
print('ADABOOST: ', 'Accuracy: ', adamodel_accuracy,'Precision: ', adamodel_precision,'Recall: ', adamodel_recall,'f1-Score: ', adamodel_f1)
acc_CV = cross_val_score(adamodel, med_features, med_class, cv=10, scoring='average_precision')
print('ADABOOST: ', acc_CV, "Mean-Precision with all Features: ", mean(acc_CV))
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
#jetzt mit XGBoost die Features bewerten und deren Anzahl reduzieren
# XGBoost
xgmodel = XGBClassifier(eval_metric='error')

xgmodel.fit(med_features_train, med_class_train)

xgmodel_pred1 = xgmodel.predict(p1235_M1)
print('XGBoost: ', xgmodel_pred1)

xgmodel_pred2 = xgmodel.predict(p3487_M1)
print('XGBoost: ', xgmodel_pred2)

xgmodel_pred3 = xgmodel.predict(p5865_M1)
print('XGBoost: ', xgmodel_pred3)

xgmodel_pred4 = xgmodel.predict(p8730_M1)
print('XGBoost: ', xgmodel_pred4)

#print(xgmodel) zeigt die Parameter des Classifiers an
xgboosted_prediction = xgmodel.predict(med_features_test)
xgboosted_accuracy = accuracy_score(med_class_test, xgboosted_prediction)
xgboosted_precision = precision_score(med_class_test, xgboosted_prediction, average='macro')
xgboosted_recall = recall_score(med_class_test, xgboosted_prediction, average='macro')
xgboosted_f1 = f1_score(med_class_test, xgboosted_prediction, average='macro')
print('XGBOOST: ', 'Accuracy: ', xgboosted_accuracy, 'Precision: ', xgboosted_precision, 'Recall: ', xgboosted_recall, 'F1-Score: ', xgboosted_f1)
acc_CV = cross_val_score(xgmodel, med_features, med_class, cv=10)
print(acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
# print(sorted((value, key) for (key, value) in xgmodel.get_booster().get_score(importance_type= 'gain').items()))
featureranking = sorted((value, key) for (key, value) in xgmodel.get_booster().get_score(importance_type= 'gain').items())
pyplot.rcParams['figure.figsize'] = [25,25]
plot_importance(xgmodel.get_booster().get_score(importance_type= 'gain'))
pyplot.show()
# # ##########################################################################################################################
# # ##########################################################################################################################
# # ##########################################################################################################################
# # ##########################################################################################################################
# newfeatures = []
# for i in range(len(featureranking)):
#     if featureranking[i][0] < 1.0:
#         newfeatures.append(featureranking[i][1])
# # # print(newfeatures)

# for el in newfeatures:
#     medDataCopy.drop(el, inplace=True, axis=1)
#     p1235_M1.drop(el, inplace=True, axis=1)
#     p3487_M1.drop(el, inplace=True, axis=1)
#     p5865_M1.drop(el, inplace=True, axis=1)
#     p8730_M1.drop(el, inplace=True, axis=1)
# medDataCopy.to_csv('naive_latest_selection.csv')
# p1235_M1.to_csv('p1235_M1_selection.csv')
# p3487_M1.to_csv('p3487_M1_selection.csv')
# p5865_M1.to_csv('p5865_M1_selection.csv')
# p8730_M1.to_csv('p8730_M1_selection.csv')