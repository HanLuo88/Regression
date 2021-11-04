from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

# df = pd.read_csv('naive_latest_todesinterval_model3.csv')
# df = df.iloc[:, 1:]

# tot = df[df['status'] != 7]
# lebendig = df[df['status'] == 7]

# #Mean für lebendig
# meandict_lebendig = {}
# lebendig = lebendig.iloc[:, 2:]
# for el in lebendig.columns:
#     meandict_lebendig[el] = lebendig[el].mean()
    
# df1 = pd.DataFrame.from_dict(meandict_lebendig, orient='index')
# # df1 = df1.transpose()

# #Mean für tot
# meandict_tot = {}
# tot = tot.iloc[:, 2:]
# for el in lebendig.columns:
#     meandict_tot[el] = tot[el].mean()
    
# df2 = pd.DataFrame.from_dict(meandict_tot, orient='index')
# # df2 = df2.transpose()

# frames = [df1, df2]
# result = pd.concat(frames, axis=1)
# result.columns = ['Lebendig_Average', 'Tot_Average']
# result.to_csv('M4_means.csv')
###########################################################################################################################
# mdf = pd.read_csv('M4_means.csv')
# mdf['relative_Abweichung'] = abs((mdf['Tot_Average'] - mdf['Lebendig_Average']) / mdf['Lebendig_Average'])
# mdf.sort_values(by='relative_Abweichung', inplace=True, ascending=False)
# mdf = mdf.reset_index()
# mdf.to_csv('M4_means_Distance.csv')
###########################################################################################################################
# cols = ['Pseudonym', 'relatives_datum', 'CRP', 'GGT37', 'GOT37', 'GPT37', 'HB', 'LEUKO', 'THROMB', 'LDH37', 'M-BL', 'FERR', 'BK-PCR', 'CMV-DNA', 'EBV-DNA', 'HST', 'M-AZ', 'M-NRBC', 'M-PR', 'NRBC-ABS','IL-6', 'status']
# frames = []
# for el in cols:
#     tmpseries = df.loc[:, el]
#     frames.append(tmpseries)
# result = pd.concat(frames, axis=1)
# result.to_csv('model4.csv')
###########################################################################################################################

# df = pd.read_csv('model4.csv')
# df = df.iloc[:, 1:]
# p1235_M4 = df[df['Pseudonym'] == 1235]
# p1235_M4.drop('status', inplace=True, axis=1)
# p1235_M4.to_csv('p1235_M4.csv')

# p3487_M4 = df[df['Pseudonym'] == 3487]
# p3487_M4.drop('status', inplace=True, axis=1)
# p3487_M4.to_csv('p3487_M4.csv')

# p5865_M4 = df[df['Pseudonym'] == 5865]
# p5865_M4.drop('status', inplace=True, axis=1)
# p5865_M4.to_csv('p5865_M4.csv')

# p8730_M4 = df[df['Pseudonym'] == 8730]
# p8730_M4.drop('status', inplace=True, axis=1)
# p8730_M4.to_csv('p8730_M4.csv')

# p124_M4 = df[df['Pseudonym'] == 124]
# p124_M4.drop('status', inplace=True, axis=1)
# p124_M4.to_csv('p124_M4.csv')


# p3297_M4 = df[df['Pseudonym'] == 3297]
# p3297_M4.drop('status', inplace=True, axis=1)
# p3297_M4.to_csv('p3297_M4.csv')


# p6658_M4 = df[df['Pseudonym'] == 6658]
# p6658_M4.drop('status', inplace=True, axis=1)
# p6658_M4.to_csv('p6658_M4.csv')


# p282441_M4 = df[df['Pseudonym'] == 282441]
# p282441_M4.drop('status', inplace=True, axis=1)
# p282441_M4.to_csv('p282441_M4.csv')


# df2 = df[(df['Pseudonym'] != 1235) & (df['Pseudonym'] != 3487) & (df['Pseudonym'] != 5865) & (df['Pseudonym'] != 8730) & (df['Pseudonym'] != 124) & (df['Pseudonym'] != 3297) & (df['Pseudonym'] != 6658) & (df['Pseudonym'] != 282441)]
# df2.to_csv('model4_TMP.csv')

