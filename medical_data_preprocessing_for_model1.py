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
#############################################################################################
#Zeile löschen, die keine Untersuchungsart beinhaltet
completeDFcopy['ABKU'].replace('', np.nan, inplace=True)
completeDFcopy.dropna(subset=['ABKU'], inplace=True)
#############################################################################################
distinct_ABKU = complete_dataDF['ABKU'].unique()
distinct_ABKUlist = distinct_ABKU.tolist()
distinct_ABKUlist.append('Pseudonym')
distinct_ABKUlist.append('relatives_datum')
distinct_ABKUlist.append('Status')
print(distinct_ABKUlist)
distinct_ABKU = np.array(distinct_ABKUlist)
print(distinct_ABKU)
# classificationtable = pd.DataFrame(columns=distinct_ABKU)
# #############################################################################################
# classificationtableCopy = classificationtable.copy()
# #############################################################################################
# sorted_complete = completeDFcopy.sort_values(["Pseudonym", "relatives_datum", 'ABKU'], ascending = True)
# sorted_complete.to_csv('sorted_complete_copy.csv')
# sorted_complete_mat = sorted_complete.to_numpy()
# uniquePseudo = sorted_complete['Pseudonym'].unique()
# ml.create_csv(uniquePseudo, sorted_complete, 0)
#############################################################################################
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