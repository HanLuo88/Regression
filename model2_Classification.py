from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost.core import Booster
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import medical_lib as ml
import pandas as pd
from statistics import mean, mode
import numpy as np
import math
import warnings
import sys

from numpy.lib.function_base import average
from sklearn.utils import multiclass
from xgboost.sklearn import XGBRFClassifier
warnings.filterwarnings("ignore")
#################################################################################################
intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
print(intervalle)
medDatamodel2 = pd.read_csv(
    'medDataCopy_model2_Features_Selected.csv')
medDataCopy_model2 = medDatamodel2.copy()
medDataCopy_model2 = medDataCopy_model2.iloc[:, 1:]
medDataCopy_model2_Features_Selected = medDataCopy_model2.copy()

#################################################################################################
med_class_model2 = medDataCopy_model2.iloc[:, -1]

med_features_model2 = medDataCopy_model2.iloc[:, :-1]

###########################################################################################################################
###########################################################################################################################
# Aufteilen der Daten in 4 Untersets
med_features_train_model2, med_features_test_model2, med_class_train_model2, med_class_test_model2 = train_test_split(
    med_features_model2, med_class_model2, test_size=0.2, random_state=43, stratify=med_class_model2)
med_class_test_array = np.array(med_class_test_model2)

p1235_M2 = pd.read_csv('p1235_M2_selection.csv')
p1235_M2 = p1235_M2.iloc[:, 1:]
print(p1235_M2.columns)
p3487_M2 = pd.read_csv('p3487_M2_selection.csv')
p3487_M2 = p3487_M2.iloc[:, 1:]
p5865_M2 = pd.read_csv('p5865_M2_selection.csv')
p5865_M2 = p5865_M2.iloc[:, 1:]
p8730_M2 = pd.read_csv('p8730_M2_selection.csv')
p8730_M2 = p8730_M2.iloc[:, 1:]

p124_M2 = pd.read_csv('p124_M2_selection.csv')
p124_M2 = p124_M2.iloc[:, 1:]
p3297_M2 = pd.read_csv('p3297_M2_selection.csv')
p3297_M2 = p3297_M2.iloc[:, 1:]
p6658_M2 = pd.read_csv('p6658_M2_selection.csv')
p6658_M2 = p6658_M2.iloc[:, 1:]
p282441_M2 = pd.read_csv('p282441_M2_selection.csv')
p282441_M2 = p282441_M2.iloc[:, 1:]

# #Ohne Boost
# print('Nachfolgend sind alle Vorhersagen ohne Featureboost')
# # Knn-Classifier for K = 8
print('KNN up to K = 4')
print('')
# for k in range(1,81):
medKNN = KNeighborsClassifier(n_neighbors=4)
# Training
medKNN.fit(med_features_train_model2, med_class_train_model2)

medKNN_pred1 = medKNN.predict(p1235_M2)
print('KNN 1235: ', medKNN_pred1)
medKNN_pred2 = medKNN.predict(p3487_M2)
print('KNN 3487: ', medKNN_pred2)
medKNN_pred3 = medKNN.predict(p5865_M2)
print('KNN 5865: ', medKNN_pred3)
medKNN_pred4 = medKNN.predict(p8730_M2)
print('KNN 8730: ', medKNN_pred4)

medKNN_pred5 = medKNN.predict(p124_M2)
print('KNN 124: ', medKNN_pred5)
medKNN_pred6 = medKNN.predict(p3297_M2)
print('KNN 3297: ', medKNN_pred6)
medKNN_pred7 = medKNN.predict(p6658_M2)
print('KNN 6658: ', medKNN_pred7)
medKNN_pred8 = medKNN.predict(p282441_M2)
print('KNN 282441: ', medKNN_pred8)
print('')
knnYpred = medKNN.predict(med_features_test_model2)
accuracyKNN = accuracy_score(knnYpred, med_class_test_array)
precisionKNN = precision_score(knnYpred, med_class_test_array, average='macro')
recallKNN = recall_score(knnYpred, med_class_test_array, average='macro')
f1scoreKNN = f1_score(knnYpred, med_class_test_array, average='macro')
print('K: ', 4, 'KNN Accuracy: ', accuracyKNN, 'KNN Precision: ',
      precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN)
print('#################################################################################################')


# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # # Jetzt mit n-fold Cross-Validation
# print('10-Fold KNN')
# print('')
# medical_KFOLD_KNN = KNeighborsClassifier(n_neighbors=4)
# # Training mit n-foldaccuracyKNN_CV, precisionKNN_CV, recallKNN_CV, f1scoreKNN_CV
# accuracyKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# precisionKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='precision')
# recallKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='recall')
# f1scoreKNN_CV = cross_val_score(medical_KFOLD_KNN, med_features_model2, med_class_model2, cv=10, scoring='f1')
# meanAccuracyKNN_CV = np.mean(accuracyKNN_CV)
# meanPrecisionKNN_CV = np.mean(precisionKNN_CV)
# meanRecallKNN_CV = np.mean(recallKNN_CV)
# meanF1scoreKNN_CV = np.mean(f1scoreKNN_CV)
# print('10-Fold KNN Accuracy: ', meanAccuracyKNN_CV, '10-Fold KNN Precision: ', meanPrecisionKNN_CV, '10-Fold KNN Recall: ', meanRecallKNN_CV, '10-Fold KNN F1-Score: ', meanF1scoreKNN_CV )
# print('#################################################################################################')

# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################

# # # # Logistic Regression:
print('Logistic Regression')
print('')
lr_model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
lr_model.fit(med_features_train_model2, med_class_train_model2)

lr_y_pred1 = lr_model.predict(p1235_M2)
print('Logistic Regression 1235: ', lr_y_pred1)
lr_y_pred2 = lr_model.predict(p3487_M2)
print('Logistic Regression 3487: ', lr_y_pred2)
lr_y_pred3 = lr_model.predict(p5865_M2)
print('Logistic Regression 5865: ', lr_y_pred3)
lr_y_pred4 = lr_model.predict(p8730_M2)
print('Logistic Regression 8730: ', lr_y_pred4)

lr_y_pred5 = lr_model.predict(p124_M2)
print('Logistic Regression 124: ', lr_y_pred5)
lr_y_pred6 = lr_model.predict(p3297_M2)
print('Logistic Regression 3297: ', lr_y_pred6)
lr_y_pred7 = lr_model.predict(p6658_M2)
print('Logistic Regression 6658: ', lr_y_pred7)
lr_y_pred8 = lr_model.predict(p282441_M2)
print('Logistic Regression 282441: ', lr_y_pred8)
print('')
# #Prediction of test set
lr_y_pred = lr_model.predict(med_features_test_model2)
lr_accuracyLogReg = accuracy_score(lr_y_pred, med_class_test_array)
lr_precisionLogReg = precision_score(
    lr_y_pred, med_class_test_array, average='macro')
lr_recallLogReg = recall_score(
    lr_y_pred, med_class_test_array, average='macro')
lr_f1scoreLogReg = f1_score(lr_y_pred, med_class_test_array, average='macro')
print('Log-Regression Accuracy: ', lr_accuracyLogReg, 'Log-Regression Precision: ', lr_precisionLogReg,
      'Log-Regression Recall: ', lr_recallLogReg, 'Log-Regression F1-Score: ', lr_f1scoreLogReg)
print('#################################################################################################')
# # # # # 10-Fold Logistic Regression:
# print('10-Fold Logistic Regression')
# print('')
# medical_KFOLD_LogReg = LogisticRegression()
# accuracyLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# precisionLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='precision')
# recallLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='recall')
# f1scoreLogReg_CV = cross_val_score(medical_KFOLD_LogReg, med_features_model2, med_class_model2, cv=10, scoring='f1')
# meanAccuracyLogReg_CV = np.mean(accuracyLogReg_CV)
# meanPrecisionLogReg_CV = np.mean(precisionLogReg_CV)
# meanRecallLogReg_CV = np.mean(recallLogReg_CV)
# meanF1scoreLogReg_CV = np.mean(f1scoreLogReg_CV)
# print('10-Fold LogReg Accuracy: ', meanAccuracyLogReg_CV, '10-Fold LogReg Precision: ', meanPrecisionLogReg_CV, '10-Fold LogReg Recall: ', meanRecallLogReg_CV, '10-Fold LogReg F1-Score: ', meanF1scoreLogReg_CV )
# print('#################################################################################################')

# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # # # SVM:
# print('SVM')
# print('')
# # # Create a svm Classifier
# medical_SVM = svm.SVC(kernel='poly', degree=9, C=1, decision_function_shape='ovo') # Linear Kernel
# #Train the model using the training sets
# medical_SVM.fit(med_features_train_model2, med_class_train_model2)

# SVM_pred1 = medical_SVM.predict(p1235)
# print('SVM: ', SVM_pred1)


# SVM_pred2 = medical_SVM.predict(p3487)
# print('SVM: ', SVM_pred2)


# SVM_pred3 = medical_SVM.predict(p5865)
# print('SVM: ', SVM_pred3)


# SVM_pred4 = medical_SVM.predict(p8730)
# print('SVM: ', SVM_pred4)
# print('')


# # #Predict the response for test dataset
# svmPred = medical_SVM.predict(med_features_test_model2)
# print('SVM: ', 'predicted: \n', lr_y_pred)
# print('SVM: ', 'Actual: \n', med_class_test_model2.to_numpy())
# accuracySVM = accuracy_score(svmPred, med_class_test_array)
# precisionSVM = precision_score(svmPred, med_class_test_array, average='macro')
# recallSVM = recall_score(svmPred, med_class_test_array, average='macro')
# f1scoreSVM = f1_score(svmPred, med_class_test_array, average='macro')
# print('SVM Accuracy: ', accuracySVM, 'SVM Precision: ', precisionSVM, 'SVM Recall: ', recallSVM, 'SVM F1-Score: ', f1scoreSVM )
# print('#################################################################################################')
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # #10-Fold SVM
# print('10-Fold SVM')
# print('')
# medical_KFOLD_SVM = svm.SVC()
# accuracySVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# precisionSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='precision')
# recallSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='recall')
# f1scoreSVM_CV = cross_val_score(medical_KFOLD_SVM, med_features_model2, med_class_model2, cv=10, scoring='f1')
# meanAccuracySVM_CV = np.mean(accuracySVM_CV)
# meanPrecisionSVM_CV = np.mean(precisionSVM_CV)
# meanRecallSVM_CV = np.mean(recallSVM_CV)
# meanF1scoreSVM_CV = np.mean(f1scoreSVM_CV)
# print('10-Fold SVM Accuracy: ', meanAccuracySVM_CV, '10-Fold SVM Precision: ', meanPrecisionSVM_CV, '10-Fold SVM Recall: ', meanRecallSVM_CV, '10-Fold SVM F1-Score: ', meanF1scoreSVM_CV )
# print('#################################################################################################')

# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # ##########################################################################################################################
# # # #Decision Tree
print('Decision Tree')
print('')
medical_DecTree = DecisionTreeClassifier(criterion='entropy')
medical_DecTree = medical_DecTree.fit(
    med_features_train_model2, med_class_train_model2)

decTree_pred1 = medical_DecTree.predict(p1235_M2)
print('Decision Tree 1235: ', decTree_pred1)
decTree_pred2 = medical_DecTree.predict(p3487_M2)
print('Decision Tree 3487: ', decTree_pred2)
decTree_pred3 = medical_DecTree.predict(p5865_M2)
print('Decision Tree 5865: ', decTree_pred3)
decTree_pred4 = medical_DecTree.predict(p8730_M2)
print('Decision Tree 8730: ', decTree_pred4)

decTree_pred5 = medical_DecTree.predict(p124_M2)
print('Decision Tree 124: ', decTree_pred5)
decTree_pred6 = medical_DecTree.predict(p3297_M2)
print('Decision Tree 3297: ', decTree_pred6)
decTree_pred7 = medical_DecTree.predict(p6658_M2)
print('Decision Tree 6658: ', decTree_pred7)
decTree_pred8 = medical_DecTree.predict(p282441_M2)
print('Decision Tree 282441: ', decTree_pred8)
print('')
decTree_pred = medical_DecTree.predict(med_features_test_model2)
accuracyDecTree = accuracy_score(decTree_pred, med_class_test_array)
precisionDecTree = precision_score(
    decTree_pred, med_class_test_array, average='macro')
recallDecTree = recall_score(
    decTree_pred, med_class_test_array, average='macro')
f1scoreDecTree = f1_score(decTree_pred, med_class_test_array, average='macro')
print('Decision Tree: ', 'medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ',
      precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree)
# print('#################################################################################################')

# # # # #10-Fold Decision Tree
# print('10-Fold Decision Tree')
# print('')
# medical_KFOLD_DecTree = DecisionTreeClassifier()
# accuracyDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# precisionDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='precision')
# recallDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='recall')
# f1scoreDecTree_CV = cross_val_score(medical_KFOLD_DecTree, med_features_model2, med_class_model2, cv=10, scoring='f1')
# meanAccuracyDecTree_CV = np.mean(accuracyDecTree_CV)
# meanPrecisionDecTree_CV = np.mean(precisionDecTree_CV)
# meanRecallDecTree_CV = np.mean(recallDecTree_CV)
# meanF1scoreDecTree_CV = np.mean(f1scoreDecTree_CV)
# print('10-Fold DecTree Accuracy: ', meanAccuracyDecTree_CV, '10-Fold DecTree Precision: ', meanPrecisionDecTree_CV, '10-Fold DecTree Recall: ', meanRecallDecTree_CV, '10-Fold DecTree F1-Score: ', meanF1scoreDecTree_CV )
# print('#################################################################################################')

# # # # ##########################################################################################################################
# # # # ##########################################################################################################################
# # # # ##########################################################################################################################
# # # # ##########################################################################################################################
# # # # # Random Forest
print('Random Forest')
print('')
# for estimator in range(50, 501, 25):
medical_RF = RandomForestClassifier(
    n_estimators=50, criterion='entropy', class_weight='balanced_subsample')
medical_RF.fit(med_features_train_model2, med_class_train_model2)

RandomForest_prediction1 = medical_RF.predict(p1235_M2)
print('Random Forest 1235: ', RandomForest_prediction1)
RandomForest_prediction2 = medical_RF.predict(p3487_M2)
print('Random Forest 3487: ', RandomForest_prediction2)
RandomForest_prediction3 = medical_RF.predict(p5865_M2)
print('Random Forest 5865: ', RandomForest_prediction3)
RandomForest_prediction4 = medical_RF.predict(p8730_M2)
print('Random Forest 8730: ', RandomForest_prediction4)

RandomForest_pred5 = medical_RF.predict(p124_M2)
print('Random Forest 124: ', RandomForest_pred5)
RandomForest_pred6 = medical_RF.predict(p3297_M2)
print('Random Forest 3297: ', RandomForest_pred6)
RandomForest_pred7 = medical_RF.predict(p6658_M2)
print('Random Forest 6658: ', RandomForest_pred7)
RandomForest_pred8 = medical_RF.predict(p282441_M2)
print('Random Forest 282441: ', RandomForest_pred8)
print('')

rfPred = medical_RF.predict(med_features_test_model2)
accuracyRF = accuracy_score(rfPred, med_class_test_array)
precisionRF = precision_score(rfPred, med_class_test_array, average='macro')
recallRF = recall_score(rfPred, med_class_test_array, average='macro')
f1scoreRF = f1_score(rfPred, med_class_test_array, average='macro')
print('Anzahl Estimator: 50 ', 'RF Accuracy: ', accuracyRF, 'RF Precision: ',
      precisionRF, 'RF Recall: ', recallRF, 'RF F1-Score: ', f1scoreRF)
print('#################################################################################################')

# # # # #10-Fold Random Forest
# print('10-Fold Random Forest')
# print('')
# medical_KFOLD_RF = RandomForestClassifier(n_estimators= 50)
# accuracyRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='accuracy')
# precisionRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='precision') #scoring='average_precision'
# recallRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='recall')
# f1scoreRF_CV = cross_val_score(medical_KFOLD_RF, med_features_model2, med_class_model2, cv=10, scoring='f1')
# meanAccuracyRF_CV = np.mean(accuracyRF_CV)
# meanPrecisionRF_CV = np.mean(precisionRF_CV)
# meanRecallRFCV = np.mean(recallRF_CV)
# meanF1scoreRF_CV = np.mean(f1scoreRF_CV)
# print('10-Fold RF Accuracy: ', meanAccuracyRF_CV, '10-Fold RF Precision: ', meanPrecisionRF_CV, '10-Fold RF Recall: ', meanRecallRFCV, '10-Fold RF F1-Score: ', meanF1scoreRF_CV )
# print('#################################################################################################')

# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # ###########################################################################################################################
# # # # # AdaBoost zum Klassifizieren
print('ADABOOST: ')
print('')
adamodel = AdaBoostClassifier()
adamodel.fit(med_features_train_model2, med_class_train_model2)

adamodel_prediction1 = adamodel.predict(p1235_M2)
print('AdaBoost 1235: ', adamodel_prediction1)
adamodel_prediction2 = adamodel.predict(p3487_M2)
print('AdaBoost 3487: ', adamodel_prediction2)
adamodel_prediction3 = adamodel.predict(p5865_M2)
print('AdaBoost 5865: ', adamodel_prediction3)
adamodel_prediction4 = adamodel.predict(p8730_M2)
print('AdaBoost 8730: ', adamodel_prediction4)

adamodel_pred5 = adamodel.predict(p124_M2)
print('AdaBoost 124: ', adamodel_pred5)
adamodel_pred6 = adamodel.predict(p3297_M2)
print('AdaBoost 3297: ', adamodel_pred6)
adamodel_pred7 = adamodel.predict(p6658_M2)
print('AdaBoost 6658: ', adamodel_pred7)
adamodel_pred8 = adamodel.predict(p282441_M2)
print('AdaBoost 282441: ', adamodel_pred8)
print('')

# #print(adamodel) zeigt die Parameter des Classifiers an
adamodel_prediction = adamodel.predict(med_features_test_model2)
adamodel_accuracy = accuracy_score(med_class_test_model2, adamodel_prediction)
adamodel_precision = precision_score(
    med_class_test_model2, adamodel_prediction, average='macro')
adamodel_recall = recall_score(
    med_class_test_model2, adamodel_prediction, average='macro')
adamodel_f1 = f1_score(med_class_test_model2,
                       adamodel_prediction, average='macro')
print('ADABOOST: ', 'Accuracy: ', adamodel_accuracy, 'Precision: ',
      adamodel_precision, 'Recall: ', adamodel_recall, 'f1-Score: ', adamodel_f1)
# acc_CV = cross_val_score(adamodel, med_features_model2, med_class_model2, cv=10, scoring='average_precision')
# print('ADABOOST: ', acc_CV, "Mean-Precision with all Features: ", mean(acc_CV))
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# #Isolation Forest
# # isoforestmodel = IsolationForest(random_state=1)
# # isoforestmodel.fit(med_features_train_model2, med_class_train_model2)
# # isoforestmodelprediction0 = isoforestmodel.predict(med_features_test_model2)
# # print(isoforestmodelprediction0)

# # isoforestmodel_prediction1 = isoforestmodel.predict(p247)
# # print('IsolationForest: ', isoforestmodel_prediction1)


# # isoforestmodel_prediction2 = isoforestmodel.predict(p18425)
# # print('IsolationForest: ', isoforestmodel_prediction2)


# # isoforestmodel_prediction3 = isoforestmodel.predict(p22278)
# # print('IsolationForest: ', isoforestmodel_prediction3)


# # isoforestmodel_prediction4 = isoforestmodel.predict(p88775)
# # print('IsolationForest: ', isoforestmodel_prediction4)
# # print('')


# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# ###########################################################################################################################
# #jetzt mit XGBoost die Features bewerten und deren Anzahl reduzieren
# # XGBoost
print('XGBOOST:')
xgmodel = XGBClassifier(n_estimators=25, eval_metric='mlogloss')
xgmodel.fit(med_features_train_model2, med_class_train_model2)
# # #print(xgmodel) zeigt die Parameter des Classifiers an
xgmodel_prediction1 = xgmodel.predict(p1235_M2)
print('XGBoost 1235: ', xgmodel_prediction1)
xgmodel_prediction2 = xgmodel.predict(p3487_M2)
print('XGBoost 3487: ', xgmodel_prediction2)
xgmodel_prediction3 = xgmodel.predict(p5865_M2)
print('XGBoost 5865: ', xgmodel_prediction3)
xgmodel_prediction4 = xgmodel.predict(p8730_M2)
print('XGBoost 8730: ', xgmodel_prediction4)

xgmodel_pred5 = xgmodel.predict(p124_M2)
print('XGBoost 124: ', xgmodel_pred5)
xgmodel_pred6 = xgmodel.predict(p3297_M2)
print('XGBoost 3297: ', xgmodel_pred6)
xgmodel_pred7 = xgmodel.predict(p6658_M2)
print('XGBoost 6658: ', xgmodel_pred7)
xgmodel_pred8 = xgmodel.predict(p282441_M2)
print('XGBoost 282441: ', xgmodel_pred8)
print('')

xgboosted_prediction = xgmodel.predict(med_features_test_model2)
xgboosted_accuracy = accuracy_score(
    med_class_test_model2, xgboosted_prediction)
xgboosted_precision = precision_score(
    med_class_test_model2, xgboosted_prediction, average='macro')
xgboosted_recall = recall_score(
    med_class_test_model2, xgboosted_prediction, average='macro')
xgboosted_f1 = f1_score(med_class_test_model2,
                        xgboosted_prediction, average='macro')
print('XGBOOST: ', 'Accuracy: ', xgboosted_accuracy, 'Precision: ',
      xgboosted_precision, 'Recall: ', xgboosted_recall, 'F1-Score: ', xgboosted_f1)
# acc_CV = cross_val_score(xgmodel, med_features_model2, med_class_model2, cv=10, scoring='precision')
# print('XGBOOST: ', acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
featureranking = sorted((value, key) for (key, value) in xgmodel.get_booster().get_score(importance_type= 'gain').items())
pyplot.rcParams['figure.figsize'] = [30,30]
plot_importance(xgmodel.get_booster().get_score(importance_type= 'gain'))
pyplot.show()
# Für Feature-Selecting nehme ich alle Features mit einem gain höher als 1
# ##########################################################################################################################
# ##########################################################################################################################
# ##########################################################################################################################
# ##########################################################################################################################


# newfeatures = []
# for i in range(len(featureranking)):
#     if featureranking[i][0] < 2.0:
#         newfeatures.append(featureranking[i][1])
# # # print(newfeatures)

# for el in newfeatures:
#     medDataCopy_model2.drop(el, inplace=True, axis=1)
#     p1235_M2.drop(el, inplace=True, axis=1)
#     p3487_M2.drop(el, inplace=True, axis=1)
#     p5865_M2.drop(el, inplace=True, axis=1)
#     p8730_M2.drop(el, inplace=True, axis=1)
#     p124_M2.drop(el, inplace=True, axis=1)
#     p3297_M2.drop(el, inplace=True, axis=1)
#     p6658_M2.drop(el, inplace=True, axis=1)
#     p282441_M2.drop(el, inplace=True, axis=1)
# medDataCopy_model2.to_csv('medDataCopy_model2_Features_Selected.csv')
# p1235_M2.to_csv('p1235_M2_selection.csv')
# p3487_M2.to_csv('p3487_M2_selection.csv')
# p5865_M2.to_csv('p5865_M2_selection.csv')
# p8730_M2.to_csv('p8730_M2_selection.csv')
# p124_M2.to_csv('p124_M2_selection.csv')
# p3297_M2.to_csv('p3297_M2_selection.csv')
# p6658_M2.to_csv('p6658_M2_selection.csv')
# p282441_M2.to_csv('p282441_M2_selection.csv')