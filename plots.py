import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

# #Entferne aus der Verstorben-Liste alle Patienten deren Tod nach 1627Tage eintrat
# print('verstorbene: ', verstorbene['Pseudonym'].nunique())
# verstorbene = verstorbene[verstorbene['relatives_datum'] <= 1627]
# print('verstorbene: ', verstorbene['Pseudonym'].nunique())
# Einlesen der Daten

complete_dataDF = pd.read_csv(
    'dump_anonymisiert_bereinigt.csv', sep=';')  # Read in CSV
# Remove the first 3 columns because they have no relevance and sort by pseudonym
complete_dataDF = complete_dataDF.iloc[:, 3:].sort_values(by='Pseudonym')
remission = pd.read_csv('Remission.csv')
infection = pd.read_csv('Infections.csv')
akGVHD = pd.read_csv('acute_GVHD.csv')
chrGVHD = pd.read_csv('chronic_GVHD.csv')
verstorben = pd.read_csv('Verstorben.csv')


gesamtPseudonyms =  complete_dataDF['Pseudonym'].nunique()

sterberatemitRemission = ml.sterberate_One(remission, verstorben, complete_dataDF)
sterberateohneRemission = ml.sterberate_One(remission, verstorben, complete_dataDF, mit = False)

sterberatemitInfektion = ml.sterberate_One(infection, verstorben, complete_dataDF)
sterberateohneInfektion = ml.sterberate_One(infection, verstorben, complete_dataDF, mit = False)

sterberatemitAk = ml.sterberate_One(akGVHD, verstorben, complete_dataDF)
sterberateohneAk = ml.sterberate_One(akGVHD, verstorben, complete_dataDF, mit = False)

sterberatemitChr = ml.sterberate_One(chrGVHD, verstorben, complete_dataDF)
sterberateohneChr = ml.sterberate_One(chrGVHD, verstorben, complete_dataDF, mit = False)


# #Patienten die alle 3 Befunde haben
# threeBefund = ml.hasBefund_three(verstorben, infection, akGVHD, chrGVHD)

# gesamt3Befund = ((akGVHD['Pseudonym'].append(chrGVHD['Pseudonym'].append(infection['Pseudonym']))).unique()).size
# print('Sterberate mit allen 3 Befunden: ', threeBefund/gesamt3Befund)


# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))
# set height of bar
sterberateMit = [sterberatemitRemission, sterberatemitInfektion, sterberatemitAk, sterberatemitChr]
sterberateOhne = [sterberateohneRemission, sterberateohneInfektion, sterberateohneAk, sterberateohneChr]

# Set position of bar on X axis
br1 = np.arange(len(sterberateMit))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot

plt.bar(br1, sterberateMit, color ='darkred', width = barWidth,
        edgecolor ='grey', label ='mit Befund')
plt.bar(br2, sterberateOhne, color ='orange', width = barWidth,
        edgecolor ='grey', label ='ohne Befund')
plt.bar(br3, 0.25, color ='deepskyblue', width = barWidth,
        edgecolor ='grey', label ='Durchschnittliche Sterberate')

# Adding Xticks
plt.xlabel('Befunde', fontweight ='bold', fontsize = 15)
plt.ylabel('Sterberate', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(sterberateMit))],
        ['Remission', 'Infektion', 'akute GVHD', 'chronische GVHD'])

plt.legend()
plt.show()

