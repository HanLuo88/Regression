import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

complete_dataDF = pd.read_csv(
    'dump_anonymisiert_bereinigt.csv', sep=';')  # Read in CSV
#Copy original to avoid changing it and Remove the first 3 columns because they have no relevance and sort by pseudonym
completeDFcopy = complete_dataDF.copy()
completeDFcopy = completeDFcopy.iloc[:, 3:]
remission = pd.read_csv('Remission.csv')
infection = pd.read_csv('Infections.csv')
akGVHD = pd.read_csv('acute_GVHD.csv')
chrGVHD = pd.read_csv('chronic_GVHD.csv')
verstorben = pd.read_csv('Verstorben.csv')

#############################################################################################
#Zeile löschen, die keine Untersuchungsart beinhaltet
completeDFcopy['ABKU'].replace('', np.nan, inplace=True)
completeDFcopy.dropna(subset=['ABKU'], inplace=True)
#############################################################################################
distinct_ABKU = complete_dataDF['ABKU'].unique()
distinct_ABKUlist = distinct_ABKU.tolist()
distinct_ABKUlist[:0] = ['Pseudonym']
distinct_ABKUlist.append('Status')
distinct_ABKU = np.array(distinct_ABKUlist)
classificationtable = pd.DataFrame(columns=distinct_ABKU)
#############################################################################################
classificationtableCopy = classificationtable.copy()
#############################################################################################
sorted_complete = completeDFcopy.sort_values(["Pseudonym", "relatives_datum", 'ABKU'], ascending = True)
# sorted_complete.to_csv('sorted_complete_copy.csv')
# sorted_complete_mat = sorted_complete.to_numpy()
# uniquePseudo = sorted_complete['Pseudonym'].unique()
# ml.create_csv(uniquePseudo, sorted_complete, 0)
#############################################################################################
# onlySoEzero = sorted_complete.loc[sorted_complete['relatives_datum'] <= 0] #nimm nur Datum bis Tag 0
# onlySoEzero.to_csv('sorted_complete_SoEzero.csv')
# preclas = ml.werteuebertragen(sorted_complete, classificationtableCopy)
# preclas.to_csv('preClassificationtable.csv')
# ############################################################################################
# objwithoutNumbers, dfwithoutstrcolumns = ml.removeColswithoutNumber('preClassificationtable.csv')
# objwithoutNumbers.to_csv('dfNonNumberOnly.csv')
# dfwithoutstrcolumns.to_csv('dfNumbersOnly.csv')
# ############################################################################################
# # Status hinzufügen
# tot = pd.read_csv('Verstorben.csv')
# preclassData = pd.read_csv('dfNumbersOnly.csv')
# preclassData['Status'] = np.nan
# totenliste = tot.loc[:, 'Pseudonym'].tolist()
# for rows in range(len(preclassData)):
#     tmpPseudo = preclassData.iloc[rows, preclassData.columns.get_loc('Pseudonym')]
#     if totenliste.__contains__(tmpPseudo):
#         preclassData.iloc[rows, preclassData.columns.get_loc('Status')] = 1 #1 = tot
#     else:
#         preclassData.iloc[rows, preclassData.columns.get_loc('Status')] = 0
# preclassData.to_csv('dfNumbersOnly.csv')

# ############################################################################################

# pseudonyme = sorted_complete['Pseudonym'].unique()
# patient0 = ml.ABKUhaeufigkeit(pseudonyme[0])
# patient0.to_csv('0ABKUHaeufigkeit')
# print(patient0.head())

#############################################################################################
# test = ml.polReg('0')
# ml.crea('0.csv')

