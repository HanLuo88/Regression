import unittest
import pandas as pd
import math
import warnings
import sys

from numpy.lib.function_base import average
from sklearn.utils import multiclass
from xgboost.sklearn import XGBRFClassifier
warnings.filterwarnings("ignore")
import numpy as np
from statistics import mean, mode
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

class TestStringMethods(unittest.TestCase):

    def test_feature_by_Distance(self):
        automatic = True
        automaticDistance = 0.3
        inputFile = 'naive_latest_todesinterval_model3.csv'
        distanceFile = 'M4_means_Distance.csv'
        outputFile = 'model4.csv'

        df = pd.read_csv(inputFile)
        df = df.iloc[:, 1:]

        cols = ['Pseudonym', 'relatives_datum', 'CRP', 'GGT37', 'GOT37', 'GPT37', 'HB', 'LEUKO', 'THROMB', 'LDH37',
                'M-BL', 'FERR', 'BK-PCR', 'CMV-DNA', 'EBV-DNA', 'HST', 'M-AZ', 'M-NRBC', 'M-PR', 'NRBC-ABS', 'IL-6', 'status']

        input = pd.read_csv(distanceFile)
        tmp = input[input['relative_Abweichung'] >= automaticDistance]
        tmpcols = ['Pseudonym', 'relatives_datum']
        for el in tmp['Unnamed: 0.1']:
            print(el)
            tmpcols.append(el)
        tmpcols.append('status')

        if automatic:
            cols = tmpcols

        frames = []
        for el in cols:
            tmpseries = df.loc[:, el]
            frames.append(tmpseries)
        result = pd.concat(frames, axis=1)
        result.to_csv(outputFile)

    def test_remove_verificationPatients(self):
        inputFile = 'model4.csv'
        outputFile = 'model4_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M4 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M4.drop('status', inplace=True, axis=1)
            tot_M4.to_csv('p' + str(toRemoveDead[index]) + '_M4.csv')

            lebend_M4 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M4.drop('status', inplace=True, axis=1)
            lebend_M4.to_csv('p' + str(toRemoveSurvivor[index]) + '_M4.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)

    def test_model4_Classification(self):
        notBoosted = True
        if notBoosted:
            inputdataset = 'model4_TMP.csv'

            p1235_M4 = pd.read_csv('p1235_M4.csv')
            p1235_M4 = p1235_M4.iloc[:, 1:]
            print(p1235_M4.columns)
            p3487_M4 = pd.read_csv('p3487_M4.csv')
            p3487_M4 = p3487_M4.iloc[:, 1:]
            p5865_M4 = pd.read_csv('p5865_M4.csv')
            p5865_M4 = p5865_M4.iloc[:, 1:]
            p8730_M4 = pd.read_csv('p8730_M4.csv')
            p8730_M4 = p8730_M4.iloc[:, 1:]

            p124_M4 = pd.read_csv('p124_M4.csv')
            p124_M4 = p124_M4.iloc[:, 1:]
            p3297_M4 = pd.read_csv('p3297_M4.csv')
            p3297_M4 = p3297_M4.iloc[:, 1:]
            p6658_M4 = pd.read_csv('p6658_M4.csv')
            p6658_M4 = p6658_M4.iloc[:, 1:]
            p282441_M4 = pd.read_csv('p282441_M4.csv')
            p282441_M4 = p282441_M4.iloc[:, 1:]
        else:
            inputdataset = 'model4_Selected.csv'

            p1235_M4 = pd.read_csv('p1235_M4_selection.csv')
            p1235_M4 = p1235_M4.iloc[:, 1:]
            print(p1235_M4.columns)
            p3487_M4 = pd.read_csv('p3487_M4_selection.csv')
            p3487_M4 = p3487_M4.iloc[:, 1:]
            p5865_M4 = pd.read_csv('p5865_M4_selection.csv')
            p5865_M4 = p5865_M4.iloc[:, 1:]
            p8730_M4 = pd.read_csv('p8730_M4_selection.csv')
            p8730_M4 = p8730_M4.iloc[:, 1:]

            p124_M4 = pd.read_csv('p124_M4_selection.csv')
            p124_M4 = p124_M4.iloc[:, 1:]
            p3297_M4 = pd.read_csv('p3297_M4_selection.csv')
            p3297_M4 = p3297_M4.iloc[:, 1:]
            p6658_M4 = pd.read_csv('p6658_M4_selection.csv')
            p6658_M4 = p6658_M4.iloc[:, 1:]
            p282441_M4 = pd.read_csv('p282441_M4_selection.csv')
            p282441_M4 = p282441_M4.iloc[:, 1:]
        medDatamodel4 = pd.read_csv(inputdataset)
        medDataCopy_model4 = medDatamodel4.copy()
        medDataCopy_model4 = medDataCopy_model4.iloc[:, 3:]
        print(medDataCopy_model4.columns)
        #################################################################################################
        med_class_model4 = medDataCopy_model4.iloc[:, -1]

        med_features_model4 = medDataCopy_model4.iloc[:, :-1]

        ###########################################################################################################################
        ###########################################################################################################################
        # Aufteilen der Daten in 4 Untersets
        med_features_train_model4, med_features_test_model4, med_class_train_model4, med_class_test_model4 = train_test_split(med_features_model4, med_class_model4, test_size=0.2, random_state=43, stratify=med_class_model4)
        med_class_test_array = np.array(med_class_test_model4)

        # print(med_features_train_model4.columns)



        p1235 = pd.read_csv('p1235_M4_selection.csv')
        p1235 = p1235.iloc[:, 3:]
        # print(p1235.columns)
        p3487 = pd.read_csv('p3487_M4_selection.csv')
        p3487 = p3487.iloc[:, 3:]
        p5865 = pd.read_csv('p5865_M4_selection.csv')
        p5865 = p5865.iloc[:, 3:]
        p8730 = pd.read_csv('p8730_M4_selection.csv')
        p8730 = p8730.iloc[:, 3:]

        p124 = pd.read_csv('p124_M4_selection.csv')
        p124 = p124.iloc[:, 3:]
        p3297 = pd.read_csv('p3297_M4_selection.csv')
        p3297 = p3297.iloc[:, 3:]
        p6658 = pd.read_csv('p6658_M4_selection.csv')
        p6658 = p6658.iloc[:, 3:]
        p282441 = pd.read_csv('p282441_M4_selection.csv')
        p282441 = p282441.iloc[:, 3:]
        intervalle = [(-520, -200),(-199, 0),(1, 120),(121, 300),(301, 800),(801, 1650)]
        print(intervalle)


        # #Ohne Boost
        # print('Nachfolgend sind alle Vorhersagen ohne Featureboost')
        # # Knn-Classifier for K = 8
        print('KNN up to K = 4')
        print('')
        # for k in range(1,81):
        medKNN = KNeighborsClassifier(n_neighbors=4)
            #Training
        medKNN.fit(med_features_train_model4,med_class_train_model4)

        medKNN_pred1 = medKNN.predict(p1235)
        print('KNN: ', medKNN_pred1)

        medKNN_pred2 = medKNN.predict(p3487)
        print('KNN: ', medKNN_pred2)

        medKNN_pred3 = medKNN.predict(p5865)
        print('KNN: ', medKNN_pred3)

        medKNN_pred4 = medKNN.predict(p8730)
        print('KNN: ', medKNN_pred4)

        medKNN_prediction5 = medKNN.predict(p124)
        print('KNN: ', medKNN_prediction5)


        medKNN_prediction6 = medKNN.predict(p3297)
        print('KNN: ', medKNN_prediction6)


        medKNN_prediction7 = medKNN.predict(p6658)
        print('KNN: ', medKNN_prediction7)


        medKNN_prediction8 = medKNN.predict(p282441)
        print('KNN: ', medKNN_prediction8)
        knnYpred = medKNN.predict(med_features_test_model4)
        # print('KNN: ', 'Prediction: ', knnYpred)
        # print('KNN: ', 'Actual: \n', med_class_test_model3.to_numpy())
        accuracyKNN = accuracy_score(knnYpred, med_class_test_array)
        precisionKNN = precision_score(knnYpred, med_class_test_array, average='weighted')
        recallKNN = recall_score(knnYpred, med_class_test_array, average='weighted')
        f1scoreKNN = f1_score(knnYpred, med_class_test_array, average='weighted')
        print('K: ', 4, 'KNN Accuracy: ', accuracyKNN, 'KNN Precision: ', precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN )
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
        lr_model = LogisticRegression(solver='newton-cg' ,multi_class='multinomial')
        lr_model.fit(med_features_train_model4, med_class_train_model4)

        lr_y_pred1 = lr_model.predict(p1235)
        print('Logistic Regression: ', lr_y_pred1)

        lr_y_pred2 = lr_model.predict(p3487)
        print('Logistic Regression: ', lr_y_pred2)

        lr_y_pred3 = lr_model.predict(p5865)
        print('Logistic Regression: ', lr_y_pred3)

        lr_y_pred4 = lr_model.predict(p8730)
        print('Logistic Regression: ', lr_y_pred4)

        lr_y_prediction5 = lr_model.predict(p124)
        print('Logistic Regression: ', lr_y_prediction5)


        lr_y_prediction6 = lr_model.predict(p3297)
        print('Logistic Regression: ', lr_y_prediction6)


        lr_y_prediction7 = lr_model.predict(p6658)
        print('Logistic Regression: ', lr_y_prediction7)


        lr_y_prediction8 = lr_model.predict(p282441)
        print('Logistic Regression: ', lr_y_prediction8)
        # #Prediction of test set
        lr_y_pred = lr_model.predict(med_features_test_model4)
        # print('Logistic Regression', 'predicted: \n', lr_y_pred)
        # print('Logistic Regression', 'Actual: \n', med_class_test_model3.to_numpy())

        lr_accuracyLogReg = accuracy_score(lr_y_pred, med_class_test_array)
        lr_precisionLogReg = precision_score(lr_y_pred, med_class_test_array, average='weighted')
        lr_recallLogReg = recall_score(lr_y_pred, med_class_test_array, average='weighted')
        lr_f1scoreLogReg = f1_score(lr_y_pred, med_class_test_array, average='weighted')
        print('Log-Regression Accuracy: ', lr_accuracyLogReg, 'Log-Regression Precision: ', lr_precisionLogReg, 'Log-Regression Recall: ', lr_recallLogReg, 'Log-Regression F1-Score: ', lr_f1scoreLogReg )
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
        # # # # SVM:
        # print('SVM')
        # print('')
        # # # Create a svm Classifier
        # medical_SVM = svm.SVC(kernel='linear',  decision_function_shape='ovo') # Linear Kernel
        # #Train the model using the training sets
        # medical_SVM.fit(med_features_train_model4, med_class_train_model4)

        # SVM_pred1 = medical_SVM.predict(p1235)
        # print('SVM: ', SVM_pred1)


        # SVM_pred2 = medical_SVM.predict(p3487)
        # print('SVM: ', SVM_pred2)


        # SVM_pred3 = medical_SVM.predict(p5865)
        # print('SVM: ', SVM_pred3)


        # SVM_pred4 = medical_SVM.predict(p8730)
        # print('SVM: ', SVM_pred4)


        # SVM_prediction5 = medical_SVM.predict(p124)
        # print('SVM: ', SVM_prediction5)


        # SVM_prediction6 = medical_SVM.predict(p3297)
        # print('SVM: ', SVM_prediction6)


        # SVM_prediction7 = medical_SVM.predict(p6658)
        # print('SVM: ', SVM_prediction7)


        # SVM_prediction8 = medical_SVM.predict(p282441)
        # print('SVM: ', SVM_prediction8)

        # print('')



        # # #Predict the response for test dataset
        # svmPred = medical_SVM.predict(med_features_test_model4)
        # # print('SVM: ', 'predicted: \n', lr_y_pred)
        # # print('SVM: ', 'Actual: \n', med_class_test_model4.to_numpy())
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
        medical_DecTree = medical_DecTree.fit(med_features_train_model4, med_class_train_model4)

        decTree_pred1 = medical_DecTree.predict(p1235)
        print('Decision Tree: ', decTree_pred1)


        decTree_pred2 = medical_DecTree.predict(p3487)
        print('Decision Tree: ', decTree_pred2)


        decTree_pred3 = medical_DecTree.predict(p5865)
        print('Decision Tree: ', decTree_pred3)


        decTree_pred4 = medical_DecTree.predict(p8730)
        print('Decision Tree: ', decTree_pred4)


        decTree_prediction5 = medical_DecTree.predict(p124)
        print('Decision Tree: ', decTree_prediction5)


        decTree_prediction6 = medical_DecTree.predict(p3297)
        print('Decision Tree: ', decTree_prediction6)


        decTree_prediction7 = medical_DecTree.predict(p6658)
        print('Decision Tree: ', decTree_prediction7)


        decTree_prediction8 = medical_DecTree.predict(p282441)
        print('Random Forest: ', decTree_prediction8)


        decTree_pred = medical_DecTree.predict(med_features_test_model4)
        # print('Decision Tree: ','predicted: \n', decTree_pred)
        # print('Decision Tree: ','Actual: \n', med_class_test_model3.to_numpy())

        accuracyDecTree = accuracy_score(decTree_pred, med_class_test_array)
        precisionDecTree = precision_score(decTree_pred, med_class_test_array, average='weighted')
        recallDecTree = recall_score(decTree_pred, med_class_test_array, average='weighted')
        f1scoreDecTree = f1_score(decTree_pred, med_class_test_array, average='weighted')
        print('Decision Tree: ','medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ', precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree )
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
        medical_RF = RandomForestClassifier(n_estimators= 50, criterion='entropy', class_weight='balanced_subsample')
        medical_RF.fit(med_features_train_model4, med_class_train_model4)

        RandomForest_prediction1 = medical_RF.predict(p1235)
        print('Random Forest: ', RandomForest_prediction1)


        RandomForest_prediction2 = medical_RF.predict(p3487)
        print('Random Forest: ', RandomForest_prediction2)


        RandomForest_prediction3 = medical_RF.predict(p5865)
        print('Random Forest: ', RandomForest_prediction3)


        RandomForest_prediction4 = medical_RF.predict(p8730)
        print('Random Forest: ', RandomForest_prediction4)


        RandomForest_prediction5 = medical_RF.predict(p124)
        print('Random Forest: ', RandomForest_prediction5)


        RandomForest_prediction6 = medical_RF.predict(p3297)
        print('Random Forest: ', RandomForest_prediction6)


        RandomForest_prediction7 = medical_RF.predict(p6658)
        print('Random Forest: ', RandomForest_prediction7)


        RandomForest_prediction8 = medical_RF.predict(p282441)
        print('Random Forest: ', RandomForest_prediction8)
        print('')

        rfPred = medical_RF.predict(med_features_test_model4)
        # print('Random Forest: ','predicted: \n', decTree_pred)
        # print('Random Forest: ','Actual: \n', med_class_test_model3.to_numpy())
        accuracyRF = accuracy_score(rfPred, med_class_test_array)
        precisionRF = precision_score(rfPred, med_class_test_array, average='weighted')
        recallRF = recall_score(rfPred, med_class_test_array, average='weighted')
        f1scoreRF = f1_score(rfPred, med_class_test_array, average='weighted')
        print('Anzahl Estimator: 50 ', 'RF Accuracy: ', accuracyRF, 'RF Precision: ', precisionRF, 'RF Recall: ', recallRF, 'RF F1-Score: ', f1scoreRF )
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
        adamodel.fit(med_features_train_model4, med_class_train_model4)

        adamodel_prediction1 = adamodel.predict(p1235)
        print('AdaBoost: ', adamodel_prediction1)


        adamodel_prediction2 = adamodel.predict(p3487)
        print('AdaBoost: ', adamodel_prediction2)


        adamodel_prediction3 = adamodel.predict(p5865)
        print('AdaBoost: ', adamodel_prediction3)


        adamodel_prediction4 = adamodel.predict(p8730)
        print('AdaBoost: ', adamodel_prediction4)


        adamodel_prediction5 = adamodel.predict(p124)
        print('AdaBoost: ', adamodel_prediction5)


        adamodel_prediction6 = adamodel.predict(p3297)
        print('AdaBoost: ', adamodel_prediction6)


        adamodel_prediction7 = adamodel.predict(p6658)
        print('AdaBoost: ', adamodel_prediction7)


        adamodel_prediction8 = adamodel.predict(p282441)
        print('AdaBoost: ', adamodel_prediction8)
        print('')

        # #print(adamodel) zeigt die Parameter des Classifiers an
        adamodel_prediction = adamodel.predict(med_features_test_model4)
        # print('AdaBoost: ','predicted: \n', adamodel_prediction)
        # print('AdaBoost: ','Actual: \n', med_class_test_model3.to_numpy())

        adamodel_accuracy = accuracy_score(med_class_test_model4, adamodel_prediction)
        adamodel_precision = precision_score(med_class_test_model4, adamodel_prediction, average='weighted')
        adamodel_recall = recall_score(med_class_test_model4, adamodel_prediction, average='weighted')
        adamodel_f1 = f1_score(med_class_test_model4, adamodel_prediction, average='weighted')
        print('ADABOOST: ', 'Accuracy: ', adamodel_accuracy,'Precision: ', adamodel_precision,'Recall: ', adamodel_recall,'f1-Score: ', adamodel_f1)
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
        xgmodel = XGBClassifier(n_estimators=25, eval_metric = 'mlogloss')
        xgmodel.fit(med_features_train_model4, med_class_train_model4)
        # # #print(xgmodel) zeigt die Parameter des Classifiers an


        xgmodel_prediction1 = xgmodel.predict(p1235)
        print('XGBoost: ', xgmodel_prediction1)


        xgmodel_prediction2 = xgmodel.predict(p3487)
        print('XGBoost: ', xgmodel_prediction2)


        xgmodel_prediction3 = xgmodel.predict(p5865)
        print('XGBoost: ', xgmodel_prediction3)


        xgmodel_prediction4 = xgmodel.predict(p8730)
        print('XGBoost: ', xgmodel_prediction4)



        xgmodel_prediction5 = xgmodel.predict(p124)
        print('XGBoost: ', xgmodel_prediction5)


        xgmodel_prediction6 = xgmodel.predict(p3297)
        print('XGBoost: ', xgmodel_prediction6)


        xgmodel_prediction7 = xgmodel.predict(p6658)
        print('XGBoost: ', xgmodel_prediction7)


        xgmodel_prediction8 = xgmodel.predict(p282441)
        print('XGBoost: ', xgmodel_prediction8)



        xgboosted_prediction = xgmodel.predict(med_features_test_model4)
        # print('predicted: \n', xgboosted_prediction)
        # print('Actual: \n', med_class_test_model3.to_numpy())


        xgboosted_accuracy = accuracy_score(med_class_test_model4, xgboosted_prediction)
        xgboosted_precision = precision_score(med_class_test_model4, xgboosted_prediction, average='weighted')
        xgboosted_recall = recall_score(med_class_test_model4, xgboosted_prediction, average='weighted')
        xgboosted_f1 = f1_score(med_class_test_model4, xgboosted_prediction, average='weighted')
        print('XGBOOST: ', 'Accuracy: ', xgboosted_accuracy, 'Precision: ', xgboosted_precision, 'Recall: ', xgboosted_recall, 'F1-Score: ', xgboosted_f1)
        # acc_CV = cross_val_score(xgmodel, med_features_model2, med_class_model2, cv=10, scoring='precision')
        # print('XGBOOST: ', acc_CV, "Mean-Accuracy with all Features: ", mean(acc_CV))
        featureranking = sorted((value, key) for (key, value) in xgmodel.get_booster().get_score(importance_type= 'gain').items())
        print(featureranking)
        pyplot.rcParams['figure.figsize'] = [30,30]
        plot_importance(xgmodel.get_booster().get_score(importance_type= 'gain'))
        pyplot.show()
        #F??r Feature-Selecting nehme ich alle Features mit einem gain h??her als 1
        # ##########################################################################################################################
        # ##########################################################################################################################
        # ##########################################################################################################################
        # ##########################################################################################################################


        # newfeatures = []
        # for i in range(len(featureranking)):
        #     if featureranking[i][0] < 0.4:
        #         newfeatures.append(featureranking[i][1])
        # # print(newfeatures)

        # for el in newfeatures:
        #     medDataCopy_model4.drop(el, inplace=True, axis=1)
        #     p1235.drop(el, inplace=True, axis=1)
        #     p3487.drop(el, inplace=True, axis=1)
        #     p5865.drop(el, inplace=True, axis=1)
        #     p8730.drop(el, inplace=True, axis=1)

        #     p124.drop(el, inplace=True, axis=1)
        #     p3297.drop(el, inplace=True, axis=1)
        #     p6658.drop(el, inplace=True, axis=1)
        #     p282441.drop(el, inplace=True, axis=1)
        # medDataCopy_model4.to_csv('model4_Selected.csv')
        # p1235.to_csv('p1235_M4_selection.csv')
        # p3487.to_csv('p3487_M4_selection.csv')
        # p5865.to_csv('p5865_M4_selection.csv')
        # p8730.to_csv('p8730_M4_selection.csv')

        # p124.to_csv('p124_M4_selection.csv')
        # p3297.to_csv('p3297_M4_selection.csv')
        # p6658.to_csv('p6658_M4_selection.csv')
        # p282441.to_csv('p282441_M4_selection.csv')


if __name__ == '__main__':
    unittest.main()
