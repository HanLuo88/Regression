from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]

############################################################################################################################
# Eingrenzung des Betrachtungszeitraumes:
# Model 2 teilt den Zeitraum in intervalle und nimmt pro Patient den letzten Wert in diesem Intervall pro Zeile.
# Dadurch hat jeder Patient mehr als eine Zeile am ende.
# Nach Tag 0 anhand der Verteilung der relativen Tage:  Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü + 3S = 2031.95 Tage = 2032 Tage
# Vor Tag 0 anhand der Verteilung der relativen Tage: Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü - s = -519.21 Tage = -520 Tage
# dfcomplete = pd.read_csv('transposed_complete.csv')
# dfcomplete = dfcomplete.iloc[:, 1:]
# dfcomplete = dfcomplete[(dfcomplete.relatives_datum < 1620)
#                         & (dfcomplete.relatives_datum > -520)]
# dfcomplete.reset_index(drop=True, inplace=True)
# dfcomplete.to_csv('transposed_complete_model1.csv')
# # Es sind nur etwa 10.000 Zeilen weniger als vorher, also ist diese Eingrenzung vertretbar.

# # Neues Mü = 196.93 Tage, Neues Sigma = 399.85 Tage.
# ############################################################################################################################
# #löschen der vier Patienten, die nach 1620Tagen gestorben sind
# df = pd.read_csv('transposed_complete_model1.csv')
# df = df.iloc[:, 1:]
# df2 = df[(df['Pseudonym'] != 11952) & (df['Pseudonym'] != 86166) & (df['Pseudonym'] != 90954) & (df['Pseudonym'] != 231731)]
# df2.to_csv('transposed_complete_model1.csv')
############################################################################################################################
# dfNonFilled, dfFilled = ml.removeColsNotWellFilled('transposed_complete_model1.csv', 90000)
# dfNonFilled.to_csv('notFilled.csv')
# print('Fertig')
# dfFilled.to_csv('filled.csv')
# print('Fertig')
############################################################################################################################
# dffilled1 = pd.read_csv('filled.csv')
# dffilled1 = dffilled1.iloc[:, 1:]
# # print(type(dffilled1.loc[0, 'XDSON']))
# dffilled1.replace("neg", 0.0, inplace=True)
# dffilled1 = dffilled1.applymap(func=ml.isfloat)
# dffilled1.dropna(axis=1, how='all', inplace=True)
# # print(dffilled1['XDSON'].dtypes)
# dffilled1.to_csv('filled_noStr.csv')
############################################################################################################################
# dffilled1 = pd.read_csv('filled_noStr.csv')
# dffilled1 = dffilled1.iloc[:, 1:]
# dffilled1.drop('XDSON', inplace=True, axis=1)
# dffilled1.to_csv('filled_noStr.csv')
############################################################################################################################
# df = pd.read_csv('filled_noStr.csv')
# pseudolist = df['Pseudonym'].unique()
# frames = []
# i = 0
# for name in pseudolist:
#     tmp = ml.takeLatest('filled_noStr.csv', name)
#     frames.append(tmp)
#     i += 1
#     # print(i)
# result = pd.concat(frames)
# result.to_csv('naive_latest.csv')
############################################################################################################################
# ml.addStatusModel1('naive_latest.csv', 'Verstorben.csv')
# df = read_csv('naive_latest.csv')
# df = df.iloc[:, 1:]
# for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
#     df[i].fillna(df[i].mean(),inplace=True)
# df.to_csv('naive_latest.csv')
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
# df = pd.read_csv('naive_latest.csv')
# df = df.iloc[:, 1:]

# p1235_M1 = df[df['Pseudonym'] == 1235]
# p1235_M1.drop('Status', inplace=True, axis=1)
# p1235_M1.to_csv('p1235_M1.csv')

# p3487_M1 = df[df['Pseudonym'] == 3487]
# p3487_M1.drop('Status', inplace=True, axis=1)
# p3487_M1.to_csv('p3487_M1.csv')

# p5865_M1 = df[df['Pseudonym'] == 5865]
# p5865_M1.drop('Status', inplace=True, axis=1)
# p5865_M1.to_csv('p5865_M1.csv')

# p8730_M1 = df[df['Pseudonym'] == 8730]
# p8730_M1.drop('Status', inplace=True, axis=1)
# p8730_M1.to_csv('p8730_M1.csv')

# df2 = df[(df['Pseudonym'] != 1235) & (df['Pseudonym'] != 3487) & (df['Pseudonym'] != 5865) & (df['Pseudonym'] != 8730)]
# df2.to_csv('naive_latest_TMP.csv')
