import math
import warnings

from numpy.lib.function_base import average
from sklearn.utils import multiclass
from xgboost.sklearn import XGBRFClassifier
warnings.filterwarnings("ignore")
import numpy as np
from statistics import mean, mode
import pandas as pd
import medical_lib as ml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost.core import Booster
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score
#################################################################################################
# medDatamodel2 = pd.read_csv('model2_Classificationtable_intervalstatus.csv')
# medDataCopy_model2 = medDatamodel2.copy()
# medDataCopy_model2 = medDataCopy_model2.iloc[:, 3:]


# #################################################################################################
# med_class_model2 = medDataCopy_model2.iloc[:, -1]

# med_features_model2 = medDataCopy_model2.iloc[:, :-1]

# ###########################################################################################################################
# ###########################################################################################################################
# # Aufteilen der Daten in 4 Untersets
# med_features_train_model2, med_features_test_model2, med_class_train_model2, med_class_test_model2 = train_test_split(med_features_model2, med_class_model2, test_size=0.2, random_state=43, stratify=med_class_model2)
# med_class_test_array = np.array(med_class_test_model2)
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
medDatamodel2 = pd.read_csv('model2_Classificationtable_intervalstatus_TMP.csv')
medDataCopy_model2 = medDatamodel2.copy()
medDataCopy_model2 = medDataCopy_model2.iloc[:, 3:]
medDataCopy_model2_Features_Selected = medDataCopy_model2.copy()

#################################################################################################
med_class_model2 = medDataCopy_model2.iloc[:, -1]

med_features_model2 = medDataCopy_model2.iloc[:, :-1]

###########################################################################################################################
###########################################################################################################################
# Aufteilen der Daten in 4 Untersets
med_features_train_model2, med_features_test_model2, med_class_train_model2, med_class_test_model2 = train_test_split(med_features_model2, med_class_model2, test_size=0.2, random_state=43, stratify=med_class_model2)
med_class_test_array = np.array(med_class_test_model2)

p247 = pd.read_csv('p247.csv')
p247 = p247.iloc[:, 3:]
p18425 = pd.read_csv('p18425.csv')
p18425 = p18425.iloc[:, 3:]
p22278 = pd.read_csv('p22278.csv')
p22278 = p22278.iloc[:, 3:]
p88775 = pd.read_csv('p88775.csv')
p88775 = p88775.iloc[:, 3:]
# #Ohne Boost
# print('Nachfolgend sind alle Vorhersagen ohne Featureboost')
# # Knn-Classifier for K = 8
# print('KNN up to K = 100')
# print('')
# for k in range(1,81):
    
#     medKNN = KNeighborsClassifier(n_neighbors=k)
#     #Training
#     medKNN.fit(med_features_train_model2,med_class_train_model2)
#     knnYpred = medKNN.predict(med_features_test_model2)
#     accuracyKNN = accuracy_score(knnYpred, med_class_test_array)
#     precisionKNN = precision_score(knnYpred, med_class_test_array, average='macro')
#     recallKNN = recall_score(knnYpred, med_class_test_array, average='macro')
#     f1scoreKNN = f1_score(knnYpred, med_class_test_array, average='macro')
#     print('K: ', k, 'KNN Accuracy: ', accuracyKNN, 'KNN Precision: ', precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN )
#     print('#################################################################################################')


# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # # Jetzt mit n-fold Cross-Validation
# # # print('10-Fold KNN')
# # # print('')
# # # medical_KFOLD_KNN = KNeighborsClassifier(n_neighbors=7)
# # # # Training mit n-foldaccuracyKNN_CV, precisionKNN_CV, recallKNN_CV, f1scoreKNN_CV
# # # accuracyKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# # # precisionKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='precision')
# # # recallKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='recall')
# # # f1scoreKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='f1')
# # # meanAccuracyKNN_CV = np.mean(accuracyKNN_CV)
# # # meanPrecisionKNN_CV = np.mean(precisionKNN_CV)
# # # meanRecallKNN_CV = np.mean(recallKNN_CV)
# # # meanF1scoreKNN_CV = np.mean(f1scoreKNN_CV)
# # # print('10-Fold KNN Accuracy: ', meanAccuracyKNN_CV, '10-Fold KNN Precision: ', meanPrecisionKNN_CV, '10-Fold KNN Recall: ', meanRecallKNN_CV, '10-Fold KNN F1-Score: ', meanF1scoreKNN_CV )
# # # print('#################################################################################################')

# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################

# # # # Logistic Regression:
# print('Logistic Regression')
# print('')
lr_model = LogisticRegression(solver='newton-cg' ,multi_class='multinomial')
lr_model.fit(med_features_train_model2, med_class_train_model2)

lr_y_pred1 = lr_model.predict(p247)
print('Logistic Regression: ', lr_y_pred1)

lr_y_pred2 = lr_model.predict(p18425)
print('Logistic Regression: ', lr_y_pred2)

lr_y_pred3 = lr_model.predict(p22278)
print('Logistic Regression: ', lr_y_pred3)

lr_y_pred4 = lr_model.predict(p88775)
print('Logistic Regression: ', lr_y_pred4)


# #Prediction of test set
lr_y_pred = lr_model.predict(med_features_test_model2)
print('predicted: \n', lr_y_pred)
print('Actual: \n', med_class_test_model2.to_numpy())

lr_accuracyLogReg = accuracy_score(lr_y_pred, med_class_test_array)
lr_precisionLogReg = precision_score(lr_y_pred, med_class_test_array, average='macro')
lr_recallLogReg = recall_score(lr_y_pred, med_class_test_array, average='macro')
lr_f1scoreLogReg = f1_score(lr_y_pred, med_class_test_array, average='macro')
print('Log-Regression Accuracy: ', lr_accuracyLogReg, 'Log-Regression Precision: ', lr_precisionLogReg, 'Log-Regression Recall: ', lr_recallLogReg, 'Log-Regression F1-Score: ', lr_f1scoreLogReg )
print('#################################################################################################')
# # # # 10-Fold Logistic Regression:
# # # print('10-Fold Logistic Regression')
# # # print('')
# # # medical_KFOLD_LogReg = LogisticRegression()
# # # accuracyLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# # # precisionLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='precision')
# # # recallLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='recall')
# # # f1scoreLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='f1')
# # # meanAccuracyLogReg_CV = np.mean(accuracyLogReg_CV)
# # # meanPrecisionLogReg_CV = np.mean(precisionLogReg_CV)
# # # meanRecallLogReg_CV = np.mean(recallLogReg_CV)
# # # meanF1scoreLogReg_CV = np.mean(f1scoreLogReg_CV)
# # # print('10-Fold LogReg Accuracy: ', meanAccuracyLogReg_CV, '10-Fold LogReg Precision: ', meanPrecisionLogReg_CV, '10-Fold LogReg Recall: ', meanRecallLogReg_CV, '10-Fold LogReg F1-Score: ', meanF1scoreLogReg_CV )
# # # print('#################################################################################################')

# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # # # SVM:
# print('SVM')
# print('')
# # # Create a svm Classifier
# medical_SVM = svm.SVC(kernel='poly', degree=9, C=1, decision_function_shape='ovo') # Linear Kernel
# #Train the model using the training sets
# medical_SVM.fit(med_features_train_model2, med_class_train_model2)

# SVM_pred1 = medical_SVM.predict(p247)
# print('SVM: ', SVM_pred1)


# SVM_pred2 = medical_SVM.predict(p18425)
# print('SVM: ', SVM_pred2)


# SVM_pred3 = medical_SVM.predict(p22278)
# print('SVM: ', SVM_pred3)


# SVM_pred4 = medical_SVM.predict(p88775)
# print('SVM: ', SVM_pred4)
# print('')


#Predict the response for test dataset
# svmPred = medical_SVM.predict(med_features_test_model2)
# accuracySVM = accuracy_score(svmPred, med_class_test_array)
# precisionSVM = precision_score(svmPred, med_class_test_array, average='macro')
# recallSVM = recall_score(svmPred, med_class_test_array, average='macro')
# f1scoreSVM = f1_score(svmPred, med_class_test_array, average='macro')
# print('SVM Accuracy: ', accuracySVM, 'SVM Precision: ', precisionSVM, 'SVM Recall: ', recallSVM, 'SVM F1-Score: ', f1scoreSVM )
# print('#################################################################################################')
# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # #10-Fold SVM
# # # print('10-Fold SVM')
# # # print('')
# # # medical_KFOLD_SVM = svm.SVC(kernel='linear')
# # # accuracySVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# # # precisionSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='precision')
# # # recallSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='recall')
# # # f1scoreSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='f1')
# # # meanAccuracySVM_CV = np.mean(accuracySVM_CV)
# # # meanPrecisionSVM_CV = np.mean(precisionSVM_CV)
# # # meanRecallSVM_CV = np.mean(recallSVM_CV)
# # # meanF1scoreSVM_CV = np.mean(f1scoreSVM_CV)
# # # print('10-Fold SVM Accuracy: ', meanAccuracySVM_CV, '10-Fold SVM Precision: ', meanPrecisionSVM_CV, '10-Fold SVM Recall: ', meanRecallSVM_CV, '10-Fold SVM F1-Score: ', meanF1scoreSVM_CV )
# # # print('#################################################################################################')

# # ##########################################################################################################################
# # ##########################################################################################################################
# # ##########################################################################################################################
# # ##########################################################################################################################
# # #Decision Tree
print('Decision Tree')
print('')
medical_DecTree = DecisionTreeClassifier(criterion='entropy')
medical_DecTree = medical_DecTree.fit(med_features_train_model2, med_class_train_model2)

decTree_pred1 = medical_DecTree.predict(p247)
print('Decision Tree: ', decTree_pred1)


decTree_pred2 = medical_DecTree.predict(p18425)
print('Decision Tree: ', decTree_pred2)


decTree_pred3 = medical_DecTree.predict(p22278)
print('Decision Tree: ', decTree_pred3)


decTree_pred4 = medical_DecTree.predict(p88775)
print('Decision Tree: ', decTree_pred4)


decTree_pred = medical_DecTree.predict(med_features_test_model2)
print('predicted: \n', decTree_pred)
print('Actual: \n', med_class_test_model2.to_numpy())

accuracyDecTree = accuracy_score(decTree_pred, med_class_test_array)
precisionDecTree = precision_score(decTree_pred, med_class_test_array, average='macro')
recallDecTree = recall_score(decTree_pred, med_class_test_array, average='macro')
f1scoreDecTree = f1_score(decTree_pred, med_class_test_array, average='macro')
print('medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ', precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree )
# print('#################################################################################################')

# # # # #10-Fold Decision Tree
# # # print('10-Fold Decision Tree')
# # # print('')
# # # medical_KFOLD_DecTree = DecisionTreeClassifier()
# # # accuracyDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# # # precisionDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='precision')
# # # recallDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='recall')
# # # f1scoreDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='f1')
# # # meanAccuracyDecTree_CV = np.mean(accuracyDecTree_CV)
# # # meanPrecisionDecTree_CV = np.mean(precisionDecTree_CV)
# # # meanRecallDecTree_CV = np.mean(recallDecTree_CV)
# # # meanF1scoreDecTree_CV = np.mean(f1scoreDecTree_CV)
# # # print('10-Fold DecTree Accuracy: ', meanAccuracyDecTree_CV, '10-Fold DecTree Precision: ', meanPrecisionDecTree_CV, '10-Fold DecTree Recall: ', meanRecallDecTree_CV, '10-Fold DecTree F1-Score: ', meanF1scoreDecTree_CV )
# # # print('#################################################################################################')

# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # # Random Forest
# print('Random Forest')
# print('')
# # for estimator in range(50, 501, 25):
medical_RF = RandomForestClassifier(n_estimators= 50, criterion='entropy', class_weight='balanced_subsample')
medical_RF.fit(med_features_train_model2, med_class_train_model2)

RandomForest_prediction1 = medical_RF.predict(p247)
print('Random Forest: ', RandomForest_prediction1)


RandomForest_prediction2 = medical_RF.predict(p18425)
print('Random Forest: ', RandomForest_prediction2)


RandomForest_prediction3 = medical_RF.predict(p22278)
print('Random Forest: ', RandomForest_prediction3)


RandomForest_prediction4 = medical_RF.predict(p88775)
print('Random Forest: ', RandomForest_prediction4)
print('')

#     rfPred = medical_RF.predict(med_features_test_model2)
#     accuracyRF = accuracy_score(rfPred, med_class_test_array)
#     precisionRF = precision_score(rfPred, med_class_test_array, average='macro')
#     recallRF = recall_score(rfPred, med_class_test_array, average='macro')
#     f1scoreRF = f1_score(rfPred, med_class_test_array, average='macro')
#     print('Anzahl Estimator: ', estimator, 'RF Accuracy: ', accuracyRF, 'RF Precision: ', precisionRF, 'RF Recall: ', recallRF, 'RF F1-Score: ', f1scoreRF )
#     print('#################################################################################################')

# # # #10-Fold Random Forest
# # # print('10-Fold Random Forest')
# # # print('')
# # # medical_KFOLD_RF = RandomForestClassifier(n_estimators= 75)
# # # accuracyRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# # # precisionRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='average_precision')
# # # recallRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='average_recall')
# # # f1scoreRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='average_f1')
# # # meanAccuracyRF_CV = np.mean(accuracyRF_CV)
# # # meanPrecisionRF_CV = np.mean(precisionRF_CV)
# # # meanRecallRFCV = np.mean(recallRF_CV)
# # # meanF1scoreRF_CV = np.mean(f1scoreRF_CV)
# # # print('10-Fold RF Accuracy: ', meanAccuracyRF_CV, '10-Fold RF Precision: ', meanPrecisionRF_CV, '10-Fold RF Recall: ', meanRecallRFCV, '10-Fold RF F1-Score: ', meanF1scoreRF_CV )
# # # print('#################################################################################################')

# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # ###########################################################################################################################
# # # # AdaBoost zum Klassifizieren
adamodel = AdaBoostClassifier()

adamodel.fit(med_features_train_model2, med_class_train_model2)

adamodel_prediction1 = adamodel.predict(p247)
print('AdaBoost: ', adamodel_prediction1)


adamodel_prediction2 = adamodel.predict(p18425)
print('AdaBoost: ', adamodel_prediction2)


adamodel_prediction3 = adamodel.predict(p22278)
print('AdaBoost: ', adamodel_prediction3)


adamodel_prediction4 = adamodel.predict(p88775)
print('AdaBoost: ', adamodel_prediction4)
print('')

#print(xgmodel) zeigt die Parameter des Classifiers an
adamodel_prediction = adamodel.predict(med_features_test_model2)
print('predicted: \n', adamodel_prediction)
print('Actual: \n', med_class_test_model2.to_numpy())

adamodel_accuracy = accuracy_score(med_class_test_model2, adamodel_prediction)
adamodel_precision = precision_score(med_class_test_model2, adamodel_prediction, average='macro')
adamodel_recall = recall_score(med_class_test_model2, adamodel_prediction, average='macro')
adamodel_f1 = f1_score(med_class_test_model2, adamodel_prediction, average='macro')
print('Accuracy: ', adamodel_accuracy,'Precision: ', adamodel_precision,'Recall: ', adamodel_recall,'f1-Score: ', adamodel_f1)
# acc_CV = cross_val_score(adamodel, med_features_model2, med_class_model2, cv=10, scoring='average_precision')
# print(acc_CV, "Mean-Precision with all Features: ", mean(acc_CV))
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#Isolation Forest
# isoforestmodel = IsolationForest(random_state=1)
# isoforestmodel.fit(med_features_train_model2, med_class_train_model2)
# isoforestmodelprediction0 = isoforestmodel.predict(med_features_test_model2)
# print(isoforestmodelprediction0)

# isoforestmodel_prediction1 = isoforestmodel.predict(p247)
# print('IsolationForest: ', isoforestmodel_prediction1)


# isoforestmodel_prediction2 = isoforestmodel.predict(p18425)
# print('IsolationForest: ', isoforestmodel_prediction2)


# isoforestmodel_prediction3 = isoforestmodel.predict(p22278)
# print('IsolationForest: ', isoforestmodel_prediction3)


# isoforestmodel_prediction4 = isoforestmodel.predict(p88775)
# print('IsolationForest: ', isoforestmodel_prediction4)
# print('')


###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
#jetzt mit XGBoost die Features bewerten und deren Anzahl reduzieren
# XGBoost
xgmodel = XGBClassifier(n_estimators=25, eval_metric = 'mlogloss')

xgmodel.fit(med_features_train_model2, med_class_train_model2)
# #print(xgmodel) zeigt die Parameter des Classifiers an


xgmodel_prediction1 = xgmodel.predict(p247)
print('XGBoost: ', xgmodel_prediction1)


xgmodel_prediction2 = xgmodel.predict(p18425)
print('XGBoost: ', xgmodel_prediction2)


xgmodel_prediction3 = xgmodel.predict(p22278)
print('XGBoost: ', xgmodel_prediction3)


xgmodel_prediction4 = xgmodel.predict(p88775)
print('XGBoost: ', xgmodel_prediction4)

xgboosted_prediction = xgmodel.predict(med_features_test_model2)
print('predicted: \n', xgboosted_prediction)
print('Actual: \n', med_class_test_model2.to_numpy())


xgboosted_accuracy = accuracy_score(med_class_test_model2, xgboosted_prediction)
xgboosted_precision = precision_score(med_class_test_model2, xgboosted_prediction, average='macro')
xgboosted_recall = recall_score(med_class_test_model2, xgboosted_prediction, average='macro')
xgboosted_f1 = f1_score(med_class_test_model2, xgboosted_prediction, average='macro')
print('Accuracy: ', xgboosted_accuracy, 'Precision: ', xgboosted_precision, 'Recall: ', xgboosted_recall, 'F1-Score: ', xgboosted_f1)
# acc_CV = cross_val_score(xgmodel, med_features_model2, med_class_model2, cv=10, scoring='precision')
# print(acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
# featureranking = sorted((value, key) for (key, value) in xgmodel.get_booster().get_score(importance_type= 'gain').items())
# pyplot.rcParams['figure.figsize'] = [30,30]
# plot_importance(xgmodel.get_booster().get_score(importance_type= 'gain'))
# pyplot.show()
#Für Feature-Selecting nehme ich alle Features mit einem gain höher als 1
# ##########################################################################################################################
# ##########################################################################################################################
# ##########################################################################################################################
# ##########################################################################################################################
#Feature-Selection: gain >= 1

# newfeatures = []
# for i in range(len(featureranking)):
#     if featureranking[i][0] < 1:
#         newfeatures.append(featureranking[i][1])
# # print(newfeatures)

# for el in newfeatures:
#     print(el)
#     medDataCopy_model2_Features_Selected.drop(el, inplace=True, axis=1)
#     p247.drop(el, inplace=True, axis=1)
#     p18425.drop(el, inplace=True, axis=1)
#     p22278.drop(el, inplace=True, axis=1)
#     p88775.drop(el, inplace=True, axis=1)
# medDataCopy_model2_Features_Selected.to_csv('medDataCopy_model2_Features_Selected.csv')
# p247.to_csv('p247_selection.csv')
# p18425.to_csv('p18425_selection.csv')
# p22278.to_csv('p22278_selection.csv')
# p88775.to_csv('p88775_selection.csv')