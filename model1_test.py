import unittest
import pandas as pd
import medical_lib as ml
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statistics import mean, mode
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
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import math

class TestStringMethods(unittest.TestCase):

    def test_Zeitraum_Eingrenzung(self):
        intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
        # Eingrenzung des Betrachtungszeitraumes:
        # Model 2 teilt den Zeitraum in intervalle und nimmt pro Patient den letzten Wert in diesem Intervall pro Zeile.
        # Dadurch hat jeder Patient mehr als eine Zeile am ende.
        # Nach Tag 0 anhand der Verteilung der relativen Tage:  Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü + 3S = 2031.95 Tage = 2032 Tage
        # Vor Tag 0 anhand der Verteilung der relativen Tage: Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü - s = -519.21 Tage = -520 Tage
        dfcomplete = pd.read_csv('transposed_complete.csv')
        dfcomplete = dfcomplete.iloc[:, 1:]
        dfcomplete = dfcomplete[(dfcomplete.relatives_datum < 1620)
                                & (dfcomplete.relatives_datum > -520)]
        dfcomplete.reset_index(drop=True, inplace=True)
        dfcomplete.to_csv('transposed_complete_model1.csv')
        # Es sind nur etwa 10.000 Zeilen weniger als vorher, also ist diese Eingrenzung vertretbar.
############################################################################################################################
    def test_Patienten_loeschen_nach_1620_Tagen(self):
        # #löschen der vier Patienten, die nach 1620Tagen gestorben sind
        df = pd.read_csv('transposed_complete_model1.csv')
        df = df.iloc[:, 1:]
        df2 = df[(df['Pseudonym'] != 11952) & (df['Pseudonym'] != 86166) & (df['Pseudonym'] != 90954) & (df['Pseudonym'] != 231731)]
        df2.to_csv('transposed_complete_model1.csv')
############################################################################################################################    
    def test_Nicht_Gut_Gefuellte_Features_Loeschen(self):
        blanks = 90000
        notwellFilledDF = 'notFilled.csv'
        wellFilledDf = 'filled.csv'
        wellFilledNoStrings = 'filled_noStr.csv'


        #Entfernt alle nicht gut gefuellten Spalten
        dfNonFilled, dfFilled = ml.removeColsNotWellFilled('transposed_complete_model1.csv', blanks)
        dfNonFilled.to_csv(notwellFilledDF)
        # print('Fertig')
        dfFilled.to_csv(wellFilledDf)
        # print('Fertig')
        ###########################################################################################################################
        dffilled1 = pd.read_csv(wellFilledDf)
        dffilled1 = dffilled1.iloc[:, 1:]
        # print(type(dffilled1.loc[0, 'XDSON']))
        #Ersetzt 'neg' mit einer 0
        dffilled1.replace("neg", 0.0, inplace=True)
        #Prueft ob ein alphanumerischer Eintrag in ein Float umgewandelt werden kann. Falls nicht, wird es mit einem np.nan ersetzt
        dffilled1 = dffilled1.applymap(func=ml.isfloat)
        #Entfernt Spalten, die komplett aus nans Bestehen
        dffilled1.dropna(axis=1, how='all', inplace=True)
        # print(dffilled1['XDSON'].dtypes)
        dffilled1.to_csv(wellFilledNoStrings)
        ###########################################################################################################################
        dffilled2 = pd.read_csv(wellFilledNoStrings)
        dffilled2 = dffilled2.iloc[:, 1:]
        dffilled2.drop('XDSON', inplace=True, axis=1)
        dffilled2.to_csv(wellFilledNoStrings)
############################################################################################################################
    def test_Aktuellste_Werte(self):
        #Erstelle Dataframe, in der für jeden Patienten nur sein aktuellster Wert pro Feature benutzt wird
        inputDf = 'filled_noStr.csv'
        outputDf = 'naive_latest.csv'
        

        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        pseudolist = df['Pseudonym'].unique()
        frames = []
        i = 0
        for name in pseudolist:
            tmp = ml.takeLatest(inputDf, name)
            frames.append(tmp)
            i += 1
        result = pd.concat(frames)
        result.to_csv(outputDf)
############################################################################################################################
    def test_Status_setzen(self):
        inputDf = 'naive_latest.csv'
        statusDf = 'Verstorben.csv'

        #Fügt Status zur InputDf hinzu
        ml.addStatusModel1(inputDf, statusDf)

        #Füllt Spaltenweise NaNs mit dem Spaltenmittelwert
        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
            df[i].fillna(df[i].mean(),inplace=True)
        df.to_csv(inputDf)                                #Schreibt das Endergebnis wieder in inputDf
############################################################################################################################
    def test_remove_verificationPatients(self):
        inputFile = 'naive_latest.csv'
       
        outputFile = 'naive_latest_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)
        df = df.iloc[:, 1:]

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M1 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M1.drop('Status', inplace=True, axis=1)
            tot_M1.to_csv('p' + str(toRemoveDead[index]) + '_M1.csv')

            lebend_M1 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M1.drop('Status', inplace=True, axis=1)
            lebend_M1.to_csv('p' + str(toRemoveSurvivor[index]) + '_M1.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)
    
    def test_Model1_Classification(self):
    
        #################################################################################################
        ###########################################################################################################################
        ###########################################################################################################################
        notBoosted = True
        if notBoosted:
            inputdataset = 'naive_latest_TMP.csv'

            p1235_M1 = pd.read_csv('p1235_M1.csv')
            p1235_M1 = p1235_M1.iloc[:, 1:]
            print(p1235_M1.columns)
            p3487_M1 = pd.read_csv('p3487_M1.csv')
            p3487_M1 = p3487_M1.iloc[:, 1:]
            p5865_M1 = pd.read_csv('p5865_M1.csv')
            p5865_M1 = p5865_M1.iloc[:, 1:]
            p8730_M1 = pd.read_csv('p8730_M1.csv')
            p8730_M1 = p8730_M1.iloc[:, 1:]

            p124_M1 = pd.read_csv('p124_M1.csv')
            p124_M1 = p124_M1.iloc[:, 1:]
            p3297_M1 = pd.read_csv('p3297_M1.csv')
            p3297_M1 = p3297_M1.iloc[:, 1:]
            p6658_M1 = pd.read_csv('p6658_M1.csv')
            p6658_M1 = p6658_M1.iloc[:, 1:]
            p282441_M1 = pd.read_csv('p282441_M1.csv')
            p282441_M1 = p282441_M1.iloc[:, 1:]
        else:
            inputdataset = 'naive_latest_selection.csv'

            p1235_M1 = pd.read_csv('p1235_M1_selection.csv')
            p1235_M1 = p1235_M1.iloc[:, 1:]
            print(p1235_M1.columns)
            p3487_M1 = pd.read_csv('p3487_M1_selection.csv')
            p3487_M1 = p3487_M1.iloc[:, 1:]
            p5865_M1 = pd.read_csv('p5865_M1_selection.csv')
            p5865_M1 = p5865_M1.iloc[:, 1:]
            p8730_M1 = pd.read_csv('p8730_M1_selection.csv')
            p8730_M1 = p8730_M1.iloc[:, 1:]

            p124_M1 = pd.read_csv('p124_M1_selection.csv')
            p124_M1 = p124_M1.iloc[:, 1:]
            p3297_M1 = pd.read_csv('p3297_M1_selection.csv')
            p3297_M1 = p3297_M1.iloc[:, 1:]
            p6658_M1 = pd.read_csv('p6658_M1_selection.csv')
            p6658_M1 = p6658_M1.iloc[:, 1:]
            p282441_M1 = pd.read_csv('p282441_M1_selection.csv')
            p282441_M1 = p282441_M1.iloc[:, 1:]
        medData = pd.read_csv(inputdataset)
        medDataCopy = medData.copy()
        medDataCopy = medDataCopy.iloc[:, 1:]
        medDataCopy_model2_Features_Selected = medDataCopy.copy()

        #################################################################################################
        med_class = medDataCopy.iloc[:, -1]
        med_features = medDataCopy.iloc[:, :-1]
        ###########################################################################################################################
        ###########################################################################################################################
        # Aufteilen der Daten in 4 Untersets
        med_features_train, med_features_test, med_class_train, med_class_test = train_test_split(med_features, med_class, test_size=0.2, random_state=43, stratify=med_class)
       
        ###########################################################################################################################
        ###########################################################################################################################
        medKNN = KNeighborsClassifier(n_neighbors=8)
        #Training
        medKNN.fit(med_features_train,med_class_train)

        medKNN_pred1 = medKNN.predict(p1235_M1)
        print('KNN 1235: ', medKNN_pred1)
        medKNN_pred2 = medKNN.predict(p3487_M1)
        print('KNN 3487: ', medKNN_pred2)
        medKNN_pred3 = medKNN.predict(p5865_M1)
        print('KNN 5865: ', medKNN_pred3)
        medKNN_pred4 = medKNN.predict(p8730_M1)
        print('KNN 8730: ', medKNN_pred4)

        medKNN_pred5 = medKNN.predict(p124_M1)
        print('KNN 124: ', medKNN_pred5)
        medKNN_pred6 = medKNN.predict(p3297_M1)
        print('KNN 3297: ', medKNN_pred6)
        medKNN_pred7 = medKNN.predict(p6658_M1)
        print('KNN 6658: ', medKNN_pred7)
        medKNN_pred8 = medKNN.predict(p282441_M1)
        print('KNN 282441: ', medKNN_pred8)
        print('')
        knnYpred = medKNN.predict(med_features_test)
        accuracyKNN = accuracy_score(med_class_test, knnYpred)
        precisionKNN = precision_score(med_class_test, knnYpred)
        recallKNN = recall_score(med_class_test, knnYpred)
        f1scoreKNN = f1_score(med_class_test, knnYpred)
        print('KNN Accuracy: ', accuracyKNN, 'KNN Precision: ', precisionKNN, 'KNN Recall: ', recallKNN, 'KNN F1-Score: ', f1scoreKNN )
        print('#################################################################################################')
        # ###########################################################################################################################
        # ###########################################################################################################################
        # # Jetzt mit n-fold Cross-Validation
        print('10-Fold KNN')
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
        print('Logistic Regression 1235: ', lr_model_pred1)
        lr_model_pred2 = lr_model.predict(p3487_M1)
        print('Logistic Regression 3487: ', lr_model_pred2)
        lr_model_pred3 = lr_model.predict(p5865_M1)
        print('Logistic Regression 5865: ', lr_model_pred3)
        lr_model_pred4 = lr_model.predict(p8730_M1)
        print('Logistic Regression 8730: ', lr_model_pred4)

        lr_model_pred5 = lr_model.predict(p124_M1)
        print('Logistic Regression 124: ', lr_model_pred5)
        lr_model_pred6 = lr_model.predict(p3297_M1)
        print('Logistic Regression 3297: ', lr_model_pred6)
        lr_model_pred7 = lr_model.predict(p6658_M1)
        print('Logistic Regression 6658: ', lr_model_pred7)
        lr_model_pred8 = lr_model.predict(p282441_M1)
        print('Logistic Regression 282441: ', lr_model_pred8)
        print('')
        #Prediction of test set
        lr_y_pred = lr_model.predict(med_features_test)
        lr_accuracyLogReg = accuracy_score(med_class_test, lr_y_pred)
        lr_precisionLogReg = precision_score(med_class_test, lr_y_pred)
        lr_recallLogReg = recall_score(med_class_test, lr_y_pred)
        lr_f1scoreLogReg = f1_score(med_class_test, lr_y_pred)
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
        print('Decision Tree 1235: ', medical_DecTree_pred1)
        medical_DecTree_pred2 = medical_DecTree.predict(p3487_M1)
        print('Decision Tree 3487: ', medical_DecTree_pred2)
        medical_DecTree_pred3 = medical_DecTree.predict(p5865_M1)
        print('Decision Tree 5865: ', medical_DecTree_pred3)
        medical_DecTree_pred4 = medical_DecTree.predict(p8730_M1)
        print('Decision Tree 8730: ', medical_DecTree_pred4)

        medical_DecTree_pred5 = medical_DecTree.predict(p124_M1)
        print('Decision Tree 124: ', medical_DecTree_pred5)
        medical_DecTree_pred6 = medical_DecTree.predict(p3297_M1)
        print('Decision Tree 3297: ', medical_DecTree_pred6)
        medical_DecTree_pred7 = medical_DecTree.predict(p6658_M1)
        print('Decision Tree 6658: ', medical_DecTree_pred7)
        medical_DecTree_pred8 = medical_DecTree.predict(p282441_M1)
        print('Decision Tree 282441: ', medical_DecTree_pred8)
        print('')

        decTree_pred = medical_DecTree.predict(med_features_test)
        accuracyDecTree = accuracy_score(med_class_test, decTree_pred)
        precisionDecTree = precision_score(med_class_test, decTree_pred)
        recallDecTree = recall_score(med_class_test, decTree_pred)
        f1scoreDecTree = f1_score(med_class_test, decTree_pred)
        print('medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ', precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree )
        print('#################################################################################################')

        #10-Fold Decision Tree
        print('10-Fold Decision Tree')
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
        print('Random Forest, n=50')
        # for estimator in range(50, 501, 25):
        medical_RF = RandomForestClassifier(n_estimators= 50)
        medical_RF.fit(med_features_train, med_class_train)

        medical_RF_pred1 = medical_RF.predict(p1235_M1)
        print('Random Forest 1235: ', medical_RF_pred1)
        medical_RF_pred2 = medical_RF.predict(p3487_M1)
        print('Random Forest 3487: ', medical_RF_pred2)
        medical_RF_pred3 = medical_RF.predict(p5865_M1)
        print('Random Forest 5865: ', medical_RF_pred3)
        medical_RF_pred4 = medical_RF.predict(p8730_M1)
        print('Random Forest 8730: ', medical_RF_pred4)

        medical_RF_pred5 = medical_RF.predict(p124_M1)
        print('Random Forest 124: ', medical_RF_pred5)
        medical_RF_pred6 = medical_RF.predict(p3297_M1)
        print('Random Forest 3297: ', medical_RF_pred6)
        medical_RF_pred7 = medical_RF.predict(p6658_M1)
        print('KRandom ForestNN 6658: ', medical_RF_pred7)
        medical_RF_pred8 = medical_RF.predict(p282441_M1)
        print('Random Forest 282441: ', medical_RF_pred8)
        print('')

        rfPred = medical_RF.predict(med_features_test)
        accuracyRF = accuracy_score(med_class_test, rfPred)
        precisionRF = precision_score(med_class_test, rfPred)
        recallRF = recall_score(med_class_test, rfPred)
        f1scoreRF = f1_score(med_class_test, rfPred)
        print('Anzahl Estimator: 50 ', 'RF Accuracy: ', accuracyRF, 'RF Precision: ', precisionRF, 'RF Recall: ', recallRF, 'RF F1-Score: ', f1scoreRF )
        print('#################################################################################################')

        # 10-Fold Random Forest
        print('10-Fold Random Forest')
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

        adamodel = AdaBoostClassifier()
        adamodel.fit(med_features_train, med_class_train)

        adamodel_prediction1 = adamodel.predict(p1235_M1)
        print('AdaBoost 1235: ', adamodel_prediction1)
        adamodel_prediction2 = adamodel.predict(p3487_M1)
        print('AdaBoost 3487: ', adamodel_prediction2)
        adamodel_prediction3 = adamodel.predict(p5865_M1)
        print('AdaBoost 5865: ', adamodel_prediction3)
        adamodel_prediction4 = adamodel.predict(p8730_M1)
        print('AdaBoost 8730: ', adamodel_prediction4)

        adamodel_pred5 = adamodel.predict(p124_M1)
        print('AdaBoost 124: ', adamodel_pred5)
        adamodel_pred6 = adamodel.predict(p3297_M1)
        print('AdaBoost 3297: ', adamodel_pred6)
        adamodel_pred7 = adamodel.predict(p6658_M1)
        print('AdaBoost 6658: ', adamodel_pred7)
        adamodel_pred8 = adamodel.predict(p282441_M1)
        print('AdaBoost 282441: ', adamodel_pred8)
        print('')


        # #print(adamodel) zeigt die Parameter des Classifiers an
        adamodel_prediction = adamodel.predict(med_features_test)
        adamodel_accuracy = accuracy_score(med_class_test, adamodel_prediction)
        adamodel_precision = precision_score(med_class_test, adamodel_prediction)
        adamodel_recall = recall_score(med_class_test, adamodel_prediction)
        adamodel_f1 = f1_score(med_class_test, adamodel_prediction)
        print('ADABOOST: ', 'Accuracy: ', adamodel_accuracy,'Precision: ', adamodel_precision,'Recall: ', adamodel_recall,'f1-Score: ', adamodel_f1)
        acc_CV = cross_val_score(adamodel, med_features, med_class, cv=10, scoring='average_precision')
        print('ADABOOST: ', acc_CV, "Mean-Precision with all Features: ", mean(acc_CV))
        print('###########################################################################################################################')
        # ###########################################################################################################################
        # ###########################################################################################################################
        # ###########################################################################################################################
        #jetzt mit XGBoost die Features bewerten und deren Anzahl reduzieren
        # XGBoost
        xgmodel = XGBClassifier(eval_metric='error')

        xgmodel.fit(med_features_train, med_class_train)

        xgmodel_pred1 = xgmodel.predict(p1235_M1)
        print('XGBoost 1235: ', xgmodel_pred1)
        xgmodel_pred2 = xgmodel.predict(p3487_M1)
        print('XGBoost 3487: ', xgmodel_pred2)
        xgmodel_pred3 = xgmodel.predict(p5865_M1)
        print('XGBoost 5865: ', xgmodel_pred3)
        xgmodel_pred4 = xgmodel.predict(p8730_M1)
        print('XGBoost 8730: ', xgmodel_pred4)

        xgmodel_pred5 = xgmodel.predict(p124_M1)
        print('XGBoost 124: ', xgmodel_pred5)
        xgmodel_pred6 = xgmodel.predict(p3297_M1)
        print('XGBoost 3297: ', xgmodel_pred6)
        xgmodel_pred7 = xgmodel.predict(p6658_M1)
        print('XGBoost 6658: ', xgmodel_pred7)
        xgmodel_pred8 = xgmodel.predict(p282441_M1)
        print('XGBoost 282441: ', xgmodel_pred8)
        print('')

        #print(xgmodel) zeigt die Parameter des Classifiers an
        xgboosted_prediction = xgmodel.predict(med_features_test)
        xgboosted_accuracy = accuracy_score(med_class_test, xgboosted_prediction)
        xgboosted_precision = precision_score(med_class_test, xgboosted_prediction)
        xgboosted_recall = recall_score(med_class_test, xgboosted_prediction)
        xgboosted_f1 = f1_score(med_class_test, xgboosted_prediction)
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
        #     p124_M1.drop(el, inplace=True, axis=1)
        #     p3297_M1.drop(el, inplace=True, axis=1)
        #     p6658_M1.drop(el, inplace=True, axis=1)
        #     p282441_M1.drop(el, inplace=True, axis=1)
        # medDataCopy.to_csv('naive_latest_selection.csv')
        # p1235_M1.to_csv('p1235_M1_selection.csv')
        # p3487_M1.to_csv('p3487_M1_selection.csv')
        # p5865_M1.to_csv('p5865_M1_selection.csv')
        # p8730_M1.to_csv('p8730_M1_selection.csv')
        # p124_M1.to_csv('p124_M1_selection.csv')
        # p3297_M1.to_csv('p3297_M1_selection.csv')
        # p6658_M1.to_csv('p6658_M1_selection.csv')
        # p282441_M1.to_csv('p282441_M1_selection.csv')



if __name__ == '__main__':
    unittest.main()