from matplotlib import pyplot as plt
from pandas.core.algorithms import mode
from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

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
# dfcomplete.to_csv('transposed_model2.csv')
# Es sind nur etwa 10.000 Zeilen weniger als vorher, also ist diese Eingrenzung vertretbar.

# Neues Mü = 196.93 Tage, Neues Sigma = 399.85 Tage.
############################################################################################################################
#löschen der vier Patienten, die nach 1620Tagen gestorben sind
# df = pd.read_csv('transposed_model2.csv')
# df = df.iloc[:, 1:]
# # indexNames = df[(df['Pseudonym'] == 11952.0) & (df['Pseudonym'] == 86166.0) & (df['Pseudonym'] == 90954.0) & (df['Pseudonym'] == 231731.0)].index # Delete these row indexes from dataFrame
# df2 = df[(df['Pseudonym'] != 11952) & (df['Pseudonym'] != 86166) & (df['Pseudonym'] != 90954) & (df['Pseudonym'] != 231731)]

# df2.to_csv('transposed_model2.csv')
############################################################################################################################


# Festlegung der Intervalle:
# Prinzip: 90Tage um Mü herum werden kleine Intervalle gewählt. Tag 0 ist der letzte Tag des Vergangenheits-Intervalls.
intervalle = [(-520, -200), (-199, 0), (1, 540), (541, 1080), (1081, 1700)]
############################################################################################################################
#Vorbereiten von transposed:model2.csv
# df = pd.read_csv('transposed_model2.csv')
# df = df.iloc[:, 1:]

# model2nonfilled, model2filled = ml.removeColsNotWellFilled('transposed_model2.csv', 90000)
# model2nonfilled.to_csv('model2notFilled.csv')
# print('Fertig')
# model2filled.to_csv('model2filled.csv')
# print('Fertig')
# ###########################################################################################################################
# model2filled = pd.read_csv('model2filled.csv')
# model2filled = model2filled.iloc[:, 1:]
# # print(type(dffilled1.loc[0, 'XDSON']))
# model2filled.replace("neg", 0.0, inplace=True)
# model2filled = model2filled.applymap(func=ml.isfloat)
# model2filled.dropna(axis=1, how='all', inplace=True)
# # print(model2filled['XDSON'].dtypes)
# model2filled.to_csv('model2filled_noStr.csv')
# ##########################################################################################################################
# model2filled = pd.read_csv('model2filled_noStr.csv')
# model2filled = model2filled.iloc[:, 1:]
# tmpPseudo = model2filled.iloc[:, 0]
# tmpDatum = model2filled.iloc[:, 1]
# df1 = model2filled.iloc[:, 2:]
# print(df1.head())
# df1 = df1.dropna(how='all')
# df1.insert(loc=0, column='relatives_datum', value=tmpDatum)
# print(df1.head())
# df1.insert(loc=0, column='Pseudonym', value=tmpPseudo)
# print(df1.head())
# df1.drop('XDSON', inplace=True, axis=1)
# df1.to_csv('model2filled_noStr.csv')

############################################################################################################################
#Pro Intervall den neusten Wert des Intervalls nehmen
# df = pd.read_csv('model2filled_noStr.csv')
# df = df.iloc[:, 1:]
# pseudo = df['Pseudonym'].unique()
# # tmp = df[(df['Pseudonym'] == 0) & (df['relatives_datum'] >= intervalle[0][0]) & (df['relatives_datum'] <= intervalle[0][1])]
# # print(tmp)
# frames = []
# for name in pseudo:
#     for el in intervalle:
#         tmpIntervalDF = df[(df['Pseudonym'] == name) & (df['relatives_datum'] >= el[0]) & (df['relatives_datum'] <= el[1])]
#         tmpdf = ml.takeLatestAsDF(tmpIntervalDF, name)
#         frames.append(tmpdf)
# result = pd.concat(frames)
# result.to_csv('model2_interval_latest.csv')
# ###########################################################################################################################
# # Pro Patient werden leere Zellen werden mit dem mean der Spalte gefüllt
# df = pd.read_csv('model2_interval_latest.csv')
# df = df.iloc[:, 1:]
# pseudo = df['Pseudonym'].unique()
# frames = []
# for name in pseudo:
#     tmpdf = df.loc[df['Pseudonym'] == name]
#     for i in tmpdf.columns[tmpdf.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
#         ser = tmpdf.loc[:, i]
#         tmpdf[i].fillna(ser.mean(),inplace=True)
#     frames.append(tmpdf)
# result = pd.concat(frames)
# result.to_csv('model2_Classificationtable_intervalstatus.csv')
# ###########################################################################################################################
# # Restliche Leere Zellen werden mit dem mean der Spalte aufgefüllt
# df = read_csv('model2_Classificationtable_intervalstatus.csv')
# df = df.iloc[:, 1:]
# for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
#         df[i].fillna(df[i].mean(),inplace=True)
# df.to_csv('model2_Classificationtable_intervalstatus.csv')
# ###########################################################################################################################

# df = pd.read_csv('model2_Classificationtable_intervalstatus.csv')
# statusDF = pd.read_csv('Verstorben_Interval.csv')
# statusDF = statusDF.iloc[:, 1:]
# status = statusDF['Pseudonym'].unique()
# df = df.iloc[:, 1:]
# df['status'] = 6.0
# index = df.index
# pseudo = df['Pseudonym'].unique()
# for row in range(len(df)):
#     name = df.loc[row, 'Pseudonym']
#     if status.__contains__(name):
#         todesint = statusDF.query('Pseudonym == ' + str(name))['todesinterval'].to_list()
#         df.loc[row, 'status'] = todesint[0]
# df.to_csv('model2_Classificationtable_intervalstatus.csv')

# df = pd.read_csv('model2_Classificationtable_intervalstatus.csv')
# df1 = df[df.isna().any(axis=1)]
# print(df1)


# ############################################################################################################################
# ###########################################################################################################################
# df = pd.read_csv('model2_Classificationtable_intervalstatus.csv')
# statusDF = pd.read_csv('Verstorben.csv')
# statusDF = statusDF.iloc[:, 1:]
# status = statusDF['Pseudonym'].unique()
# df = df.iloc[:, 1:]
# df['Status'] = np.nan
# pseudo = df['Pseudonym'].unique()
# index = df.index
# for row in range(len(df)):
#     name = df.loc[row, 'Pseudonym']
#     if status.__contains__(name):
#         condition = df["Pseudonym"] == name
#         nameindexes = index[condition]
#         pseudoindexeslist = nameindexes.tolist()
#         df.loc[row, 'Status'] = 0
#         if row == pseudoindexeslist[-1]:
#             df.loc[row, 'Status'] = 1
# df['Status'].fillna(0,inplace=True)
# df.to_csv('model2_Classificationtable_intervalstatus.csv')
# ############################################################################################################################
# ###########################################################################################################################
# # nehme Todesintervalle der toten Patienten als Klasse
# # Füge in Verstorben.csv eine Intervallspalte hinzu
# verstorben = pd.read_csv('Verstorben.csv')
# verstorben = verstorben.iloc[:, 1:]
# for row in range(len(verstorben)):
#     tmp = 1
#     todesdatum = verstorben.loc[row, 'relatives_datum']
#     for el in intervalle:
#         if (todesdatum >= el[0]) and (todesdatum <= el[1]):
#             verstorben.loc[row, 'todesinterval'] = tmp
#             break
#         tmp += 1 
# verstorben.to_csv('Verstorben_Interval.csv')
# ueberlebensinterval = len(intervalle) + 1
