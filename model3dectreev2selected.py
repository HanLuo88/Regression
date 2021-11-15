import math
import warnings
import sys

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

#################################################################################################
medDatamodel3 = pd.read_csv('naive_latest_todesinterval_model3_v2_selection.csv') #naive_latest_todesinterval_model3_v2_TMP
medDataCopy_model3 = medDatamodel3.copy()
medDataCopy_model3 = medDataCopy_model3.iloc[:, 1:]

#################################################################################################
med_class_model3 = medDataCopy_model3.iloc[:, -1]

med_features_model3 = medDataCopy_model3.iloc[:, :-1]

###########################################################################################################################
###########################################################################################################################
# Aufteilen der Daten in 4 Untersets
med_features_train_model3, med_features_test_model3, med_class_train_model3, med_class_test_model3 = train_test_split(med_features_model3, med_class_model3, test_size=0.2, random_state=43, stratify=med_class_model3)
med_class_test_array = np.array(med_class_test_model3)

p1235 = pd.read_csv('p1235_M3_v2_selection.csv')
p1235 = p1235.iloc[:, 1:]
print(p1235.columns)
p3487 = pd.read_csv('p3487_M3_v2_selection.csv')
p3487 = p3487.iloc[:, 1:]
p5865 = pd.read_csv('p5865_M3_v2_selection.csv')
p5865 = p5865.iloc[:, 1:]
p8730 = pd.read_csv('p8730_M3_v2_selection.csv')
p8730 = p8730.iloc[:, 1:]

p124 = pd.read_csv('p124_M3_v2_selection.csv')
p124 = p124.iloc[:, 1:]
p3297 = pd.read_csv('p3297_M3_v2_selection.csv')
p3297 = p3297.iloc[:, 1:]
p6658 = pd.read_csv('p6658_M3_v2_selection.csv')
p6658 = p6658.iloc[:, 1:]
p282441 = pd.read_csv('p282441_M3_v2_selection.csv')
p282441 = p282441.iloc[:, 1:]
intervalle = [(-520, -200),(-199, 0),(1, 120),(121, 300),(301, 800),(801, 1650)]
print(intervalle)

print('Decision Tree')
medical_DecTree = DecisionTreeClassifier(random_state=17)
medical_DecTree = medical_DecTree.fit(med_features_train_model3, med_class_train_model3)

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

decTree_pred = medical_DecTree.predict(med_features_test_model3)
accuracyDecTree = accuracy_score(decTree_pred, med_class_test_array)
precisionDecTree = precision_score(decTree_pred, med_class_test_array, average='weighted')
recallDecTree = recall_score(decTree_pred, med_class_test_array, average='weighted')
f1scoreDecTree = f1_score(decTree_pred, med_class_test_array, average='weighted')
print('Decision Tree: ','medical_DecTree Accuracy: ', accuracyDecTree, 'DecTree Precision: ', precisionDecTree, 'DecTree Recall: ', recallDecTree, 'DecTree F1-Score: ', f1scoreDecTree )
pred_tot_lebendigdc = []
actual_tot_lebendigdc = []
abweichungdc = []
for el in range(0, len(decTree_pred)):
    dist = abs(decTree_pred[el] - med_class_test_array[el])
    abweichungdc.append(dist)
    if decTree_pred[el] < 7:
        pred_tot_lebendigdc.append(1)
    else: 
        pred_tot_lebendigdc.append(0)
    if med_class_test_array[el] < 7:
        actual_tot_lebendigdc.append(1)
    else:
        actual_tot_lebendigdc.append(0)
accuracydc, precisiondc, recalldc, f1scoredc = ml.scoring(pred_tot_lebendigdc, actual_tot_lebendigdc)
print(pred_tot_lebendigdc)
print('')
print(actual_tot_lebendigdc)
print('Tatsächlich: ', accuracydc, precisiondc, recalldc, f1scoredc)
print('Durchschnittliche Abweichung: ', mean(abweichungdc))
print('Standartabweichung der Abweichung: ', np.std(abweichungdc))

# result = pd.read_csv('automated_algorithmen.csv')
# result = result.iloc[:, 1:]
# result.at[11, 'Decision_Tree'] = precisiondc
# result.to_csv('automated_algorithmen.csv')


fig = pyplot.figure(figsize=(50,50))
_ = tree.plot_tree(medical_DecTree, 
                   feature_names=med_features_model3.columns,  
                   class_names=str(med_class_model3),
                   filled=True)