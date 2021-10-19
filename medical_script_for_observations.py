from matplotlib import pyplot as plt
import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ml.abkuFilter('0.csv', 'LEUKO')
onlyNonNumbers = pd.read_csv('dfNonNumberOnly.csv')
onlyNumbers = pd.read_csv('dfNumbersOnly.csv')
# leuko = onlyNonNumbers.copy()
# for row in range(len(leuko)):
#     if (leuko.LEUKO[row] == '<0.1'):
#         leuko.loc[row, 'LEUKO'] = float('0.09')
# leuko['LEUKO'] = leuko['LEUKO'].astype(float)
# print(leuko.LEUKO[8])
# leuko.sort_values(by=['LEUKO'], inplace=True, ascending=False)
# leuko.to_csv('sortByLEUKO.csv')

# ml.abkuFilter('sortByLEUKO.csv', 'LEUKO')

# featToAdd = ml.replaceValues('dfNonNumberOnly.csv', 'LEUKO', 0.01)
# onlyNumbers['LEUKO'] = featToAdd
# onlyNumbers.to_csv('dfNumbersOnly.csv')
alb = onlyNonNumbers['ALB']
try:
    for el in range(len(alb)):
	    alb[el] = alb[el].astype(float)
except:
	alb[el] = np.nan

print(alb)
