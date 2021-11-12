from scipy.sparse.construct import rand
from sklearn.utils import shuffle
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

##############################################################################################################################
#Model1v2 ist model1, jedoch gibt es genauso viele Überlebende wie Tote.
# df = pd.read_csv('naive_latest.csv')
# df = df.iloc[:, 2:]

# tot = pd.read_csv('Verstorben.csv')

# uniquetote = tot['Pseudonym'].unique().tolist()
# df1 = df[df['Pseudonym'].isin(uniquetote)]

# df2 = df[~df['Pseudonym'].isin(uniquetote)]
# uniquelebende = df2['Pseudonym'].unique().tolist()

# randomLebend = shuffle(uniquelebende, random_state=1)
# lebend114 = randomLebend[0:112]
# if 124 in lebend114:
#     print('yes')
# else:
#     lebend114.append(124)
# if 282441 in lebend114:
#     print('yes')
# else:
#     lebend114.append(282441)
# beides = lebend114 + uniquetote
# print(beides)

# df3 = df[df['Pseudonym'].isin(beides)]
# df3.to_csv('naive_latest_model1v2.csv')
##############################################################################################################################
# inputFile = 'naive_latest_model1v2.csv'
       
# outputFile = 'naive_latest_model1v2_TMP.csv'

# toRemoveSurvivor = [124, 3297, 6658, 282441]
# toRemoveDead = [1235, 3487, 5865, 8730]

# df = pd.read_csv(inputFile)

# tmpDf = df.iloc[:, 1:]
# for index in range(0, len(toRemoveSurvivor)):
#     tot_M1 = df[df['Pseudonym'] == toRemoveDead[index]]
#     tot_M1.drop('Status', inplace=True, axis=1)
#     tot_M1.to_csv('p' + str(toRemoveDead[index]) + '_M1_v2.csv')

#     lebend_M1 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
#     lebend_M1.drop('Status', inplace=True, axis=1)
#     lebend_M1.to_csv('p' + str(toRemoveSurvivor[index]) + '_M1_v2.csv')
            
#     tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

# tmpDf.to_csv(outputFile)
##############################################################################################################################