from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

############################################################################################################################
# dfNonFilled, dfFilled = ml.removeColsNotWellFilled('transposed_complete.csv', 90000)
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
# print(dffilled1['XDSON'].dtypes)
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
#     print(i)
# result = pd.concat(frames)
# result.to_csv('naive_latest.csv')
############################################################################################################################
# ml.addStatusModel1('naive_latest.csv', 'Verstorben.csv')
# df = read_csv('naive_latest.csv')
# df = df.iloc[:, 1:]
# for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
#     df[i].fillna(df[i].mean(),inplace=True)
# df.to_csv('naive_latest.csv')