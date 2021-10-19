import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def hasBefund_one(df1, befundDf):
    gesDf = df1.copy()
    befDF = befundDf.copy()
    gesPseudo = gesDf['Pseudonym'].unique()
    befPseudo = befDF['Pseudonym'].unique()
    befPseudolist = befPseudo.tolist()
    mitBefund = 0
    for entryIngespseudo in range(len(gesPseudo)):
        if befPseudolist.__contains__(gesPseudo[entryIngespseudo]):
            mitBefund = mitBefund + 1
    return mitBefund

def hasBefund_three(df1, befundDf, bf2, bf3):
    gesDf = df1.copy()
    befDF = befundDf.copy()
    gesPseudo = gesDf['Pseudonym'].unique()
    befPseudo = befDF['Pseudonym'].unique()
    bf2Pseudo = bf2['Pseudonym'].unique()
    bf3Pseudo = bf3['Pseudonym'].unique()

    befPseudolist = befPseudo.tolist()
    bf2Pseudolist = bf2Pseudo.tolist()
    bf3Pseudolist = bf3Pseudo.tolist()

    mitBefund = 0
    for entryIngespseudo in range(len(gesPseudo)):
        if befPseudolist.__contains__(gesPseudo[entryIngespseudo]) and bf2Pseudolist.__contains__(gesPseudo[entryIngespseudo]) and bf3Pseudolist.__contains__(gesPseudo[entryIngespseudo]):
            mitBefund = mitBefund + 1
    return mitBefund


def makeABKUfeaturesandcreatedf(dataframe):
    tmp = dataframe.copy()
    abkufeatures = tmp['ABKU'].unique()
    df = pd.DataFrame(columns=abkufeatures)
    return df

def sterberate_One(befundDF, statusDF, gesamtDF, mit = True):
    gesamtPseudo =  gesamtDF['Pseudonym'].nunique()
    status = statusDF['Pseudonym'].nunique()

    gesamtMitbefund = befundDF['Pseudonym'].nunique()
    gesamtOhnebefund = gesamtPseudo - gesamtMitbefund

    totMitbefund = hasBefund_one(statusDF, befundDF)
    totOhneBefund = status - totMitbefund
    if mit == True:
        tm = totMitbefund/gesamtMitbefund
        return tm
    
    to = totOhneBefund/gesamtOhnebefund
    return to


def create_csv(names, df, pseudo):
    #erstellt csv-Dateien für jeden unique Pseudonym.
    #Vorraussetzung: Ein sortiertes Dataframe, aus dem die Unique Namen entnommen werden.
    filenames = []
    for pseudo in names:
        tmpdf = df[pseudo]==pseudo
        filtered_df = df[tmpdf]
        fname = str(pseudo) + '.csv'
        filenames.append(fname)
        filtered_df.to_csv(fname)
    return filenames


def werteuebertragen(inputdf, outputdf):
    outdf = outputdf.copy()
    copyIndex = 0
    pseudo = inputdf.iloc[0, inputdf.columns.get_loc('Pseudonym')] #ersten Kundenname speichern
    for rows in range(len(inputdf)): #gehe Zeile für Zeile durch
        exist = pd.isnull(inputdf.iloc[rows, inputdf.columns.get_loc('Messwert_String')])#gibt es in dieser Zeile einen Wert für Messwert_String?
        if exist == False: #wenn ja:
            tmppseudo = inputdf.iloc[rows, inputdf.columns.get_loc('Pseudonym')] #aktuellen Zeilen-Kunden zwischenspeichern
            if tmppseudo != pseudo: 
                pseudo = tmppseudo
                copyIndex += 1
            tmpLaborart = inputdf.iloc[rows, inputdf.columns.get_loc('ABKU')] #Welcher Wert für die Untersuchungsart wurde gemessen in der rows-ten Zeile?
            outdf.at[copyIndex, 'Pseudonym'] = pseudo
            wert = inputdf.iloc[rows, inputdf.columns.get_loc('Messwert_String')]
            outdf.iloc[copyIndex, outdf.columns.get_loc(tmpLaborart)] = wert
    
    return outdf


def removeColswithoutNumber(csvfile):
    pre = pd.read_csv(csvfile)
    pre = pre.iloc[:, 1:]
    precopy = pre.copy()
    dfwithObj = pd.DataFrame()
    dfwithObj['colname'] = precopy.iloc[:, 0]
    for col in range(1, len(pre.columns)):
        spalte = precopy.iloc[:, col]
        colname = precopy.columns[col]
        if (spalte.dtype != np.number) or (spalte.isna().sum() >= 100):
            dfwithObj[colname] = spalte

    colnamelist = list(dfwithObj.columns)
    precopy.drop(columns=colnamelist[1:], axis=1, inplace=True)
    return dfwithObj, precopy


def ABKUhaeufigkeit(names):
    df = pd.read_csv(str(names) + '.csv')
    dups_ABKU = df.pivot_table(columns=['ABKU'], aggfunc='size')
    dups_ABKU.to_csv('tmpdump.csv')
    dups_ABKU2 = pd.read_csv('tmpdump.csv')
    dups_ABKU2.rename(columns={'0': 'Anzahl'}, inplace=True)
    dups_ABKU2.sort_values('Anzahl', ascending=False, inplace=True)
    dups_ABKU2.to_csv(str(names) + 'ABKUHaeufigkeit.csv')
    return dups_ABKU2


def polReg(name):
    relevantABKU = pd.read_csv('dfNumbersOnly.csv')
    relAbkulist = relevantABKU.columns.values.tolist()
    relAbkulist = relAbkulist[3:-1]
    df = pd.read_csv(str(name) + '.csv')
    df.sort_values(['ABKU', 'relatives_datum'], inplace=True, ascending=False)
    for entry in relAbkulist:
        tmpdf = df[df['ABKU']==entry]
        laborartwert = tmpdf['Messwert_String']

        datum = tmpdf['relatives_datum']
        plt.plot(datum, laborartwert)
        plt.xlabel("relatives Datum")
        plt.ylabel(str(entry) + "-Wert")
        plt.show() 
            

def scoring(predicted, truevalue):
    #takes 2 numpy arrays 
    truePositiv = 0
    trueNegativ = 0
    falsePositiv = 0
    falseNegativ = 0
    for el in range(len(predicted)):
        if (predicted[el] == truevalue[el]) and (predicted[el] == 1):
            truePositiv += 1
        elif (predicted[el] == truevalue[el]) and (predicted[el] == 0):
            trueNegativ += 1
        elif (predicted[el] != truevalue[el]) and (predicted[el] == 1):
            falsePositiv += 1
        else: 
            falseNegativ += 1
    accuracy = (truePositiv + trueNegativ) / len(predicted)
    precision = truePositiv/(truePositiv+falsePositiv)
    recall = truePositiv/(truePositiv + falseNegativ)
    f1score = 2*(recall *precision)/(recall + precision)
    return accuracy, precision, recall, f1score




def crea(filename):
    df = pd.read_csv(str(filename))
    df.sort_values(['ABKU', 'relatives_datum'], inplace=True, ascending=False)
    tmpdf = df[df['ABKU']=='CREA']
    laborartwert = tmpdf['Messwert_String']
    datum = tmpdf['relatives_datum']


    

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(datum, laborartwert)
    # plt.plot(datum, laborartwert)
    # plt.xlabel("relatives Datum")
    # plt.ylabel("CREA-Wert")
    # plt.show() 
    

    
