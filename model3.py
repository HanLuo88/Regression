import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

###########################################################################################################################
#Model 3 unterscheidet sich von Model1 nur dadurch, dass im Status entweder das Todesintervall oder das Überlebensinterval steht.
#Es ist eine Kombination von Model 1 und Model 2
###########################################################################################################################
#Intervall festlegen und in Verstorben.csv übertragen
intervalle = [(-520, -200),(-199, 0),(1, 120),(121, 300),(301, 800),(801, 1650)]

# totintervalle = ml.addtoverstorben('Verstorben.csv', intervalle)
# totintervalle.to_csv('Verstorben_Interval.csv')
###########################################################################################################################
#Dataframe einlesen
# df = pd.read_csv('naive_latest.csv')
# df = df.iloc[:, 2:]
# df.drop('Status', inplace=True, axis=1)
# df.to_csv('naive_latest_M3.csv')
#Todesinterval einfügen
# df = ml.fillintervalstatus('naive_latest_M3.csv', intervalle)
# df.to_csv('naive_latest_todesinterval_model3.csv')


# df = pd.read_csv('naive_latest_todesinterval_model3.csv')
# df = df.iloc[:, 1:]
# p1235_M3 = df[df['Pseudonym'] == 1235]
# p1235_M3.drop('status', inplace=True, axis=1)
# p1235_M3.to_csv('p1235_M3.csv')

# p3487_M3 = df[df['Pseudonym'] == 3487]
# p3487_M3.drop('status', inplace=True, axis=1)
# p3487_M3.to_csv('p3487_M3.csv')

# p5865_M3 = df[df['Pseudonym'] == 5865]
# p5865_M3.drop('status', inplace=True, axis=1)
# p5865_M3.to_csv('p5865_M3.csv')

# p8730_M3 = df[df['Pseudonym'] == 8730]
# p8730_M3.drop('status', inplace=True, axis=1)
# p8730_M3.to_csv('p8730_M3.csv')

# df2 = df[(df['Pseudonym'] != 1235) & (df['Pseudonym'] != 3487) & (df['Pseudonym'] != 5865) & (df['Pseudonym'] != 8730)]
# df2.to_csv('naive_latest_model3_TMP.csv')


# df = pd.read_csv('naive_latest_todesinterval_model3.csv')
# df = df.iloc[:, 3:]
# print(df.head())
# df.to_csv('naive_latest_todesinterval_model3.csv')