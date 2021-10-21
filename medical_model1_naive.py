import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

# gesamt = pd.read_csv('dump_anonymisiert_bereinigt.csv', sep=';')
# completeDFcopy = gesamt.copy()
# completeDFcopy = gesamt.iloc[:, 3:]
# completeDFcopy.sort_values(by=["Pseudonym", "relatives_datum", 'ABKU'], inplace=True, ascending=True)

# distinct_ABKU = completeDFcopy['ABKU'].unique()
# distinct_ABKUlist = distinct_ABKU.tolist()
# distinct_ABKUlist.append('Pseudonym')
# distinct_ABKUlist.append('relatives_datum')
# distinct_ABKUlist.append('Status')
# distinct_ABKU = np.array(distinct_ABKUlist)
# transponedTable = pd.DataFrame(columns=distinct_ABKU)
# preclas = ml.werteuebertragenALL(completeDFcopy, transponedTable)
# preclas.to_csv('ttt.csv')

transponedTable = ml.datatransformation('dump_anonymisiert_bereinigt.csv')
transponedTable.to_csv('transponed_complete.csv')

