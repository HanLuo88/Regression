from matplotlib import pyplot as plt
from pandas.core.algorithms import mode
from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import linregress


##########################################################################################################################
#Model 5 benutzt als Grundlage model2filled_noStr.csv von model2, füllt die Werte aber auf eine andere Art.
# intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
# ##########################################################################################################################
# df = pd.read_csv('model2filled_noStr.csv')
# df = df.iloc[:, 1:]
# df = df.sort_values(by=["Pseudonym", "relatives_datum"], ascending = (True, False))
# pseudo = df['Pseudonym'].unique()
# frames = []

# tmpIntervalDF = df[(df['Pseudonym'] == 0) & (df['relatives_datum'] >= intervalle[0][0]) & (df['relatives_datum'] <= intervalle[0][1])]
# print(tmpIntervalDF)
# relTage_X = tmpIntervalDF['relatives_datum'].to_list()
# abkuwert_Y = tmpIntervalDF['CA'].to_list()

# relTage_X_noNan = []
# abkuwert_Y_noNan = []
# for i in range(0, len(abkuwert_Y)):
#     if np.isnan(abkuwert_Y[i]):
#         continue
#     relTage_X_noNan.append(relTage_X[i])
#     abkuwert_Y_noNan.append(abkuwert_Y[i])
# print(relTage_X_noNan)
# print(abkuwert_Y_noNan)
# b, a, r, p = linregress
# plt.scatter(relTage_X_noNan, abkuwert_Y_noNan)
# plt.show()



# for name in pseudo:
#     for el in intervalle:
#         tmpIntervalDF = df[(df['Pseudonym'] == name) & (df['relatives_datum'] >= el[0]) & (df['relatives_datum'] <= el[1])]
#         print(tmpIntervalDF)
        # tmpdf = ml.takeLatestAsDF(tmpIntervalDF, name)
        # tmpdf
        # frames.append(tmpdf)
# result = pd.concat(frames)
