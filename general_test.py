import numpy as np
from os import TMP_MAX
import unittest
import math
import pandas as pd
import medical_lib as ml
from matplotlib import colors, pyplot as plt
from scipy.stats import linregress
import warnings
warnings.filterwarnings("ignore")

# #Einige globale Parameter die die Tests benötigen

data = pd.read_csv('model2filled_noStr.csv')
data = data.iloc[:, 1:]

pseudo = 138631
feature = 'RETI-ABS'
links = 0
rechts = 120
pseudodata = data[data['Pseudonym'] == pseudo]
verlaufdata = pseudodata[pseudodata['relatives_datum'].between(links, rechts)]
xachse = verlaufdata['relatives_datum'].to_list()
yachse = verlaufdata[feature].to_list()

for i in range(len(xachse)-1, -1, -1):
    if math.isnan(yachse[i]):
        xachse.remove(xachse[i])
        yachse.remove(yachse[i])
xachse1 = xachse[(len(xachse)-3):len(xachse)]
yachse1 = yachse[(len(yachse)-3):len(yachse)]

tmpl = rechts - 30
tmpverlaufdata = pseudodata[pseudodata['relatives_datum'].between(tmpl, rechts)]
tmpxachse = tmpverlaufdata['relatives_datum'].to_list()
tmpyachse = tmpverlaufdata[feature].to_list()
for i in range(len(tmpyachse)-1, -1, -1):
    if math.isnan(tmpyachse[i]):
        tmpxachse.remove(tmpxachse[i])
        tmpyachse.remove(tmpyachse[i])
if len(tmpyachse) < 3:
    b, a, r, p, std = linregress(xachse1, yachse1)
else:
    b, a, r, p, std = linregress(tmpxachse, tmpyachse)

print(b)

plt.rcParams['figure.figsize'] = [15, 15]
plt.plot(xachse, yachse,  label=feature, linestyle='-', marker='o', linewidth=2.5, color='orangered')
plt.legend()
plt.show()
