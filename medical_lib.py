import pandas as pd
import numpy as np



import time

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
    for n in names:
        tmpdf = df['Pseudonym']==pseudo
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
            tmpLaborart = inputdf.iloc[rows, inputdf.columns.get_loc('ABKU')] 
            outdf.at[copyIndex, 'Pseudonym'] = pseudo
            wert = inputdf.iloc[rows, inputdf.columns.get_loc('Messwert_String')] #Welcher Wert für die Untersuchungsart wurde gemessen in der rows-ten Zeile?
            outdf.iloc[copyIndex, outdf.columns.get_loc(tmpLaborart)] = wert
    return outdf



def removeColsNotWellFilled(csvfile, blanknumber):
    pre = pd.read_csv(csvfile)
    pre = pre.iloc[:, 1:]
    filled = pre.copy()
    nonfilled = pd.DataFrame()
    nonfilled['Pseudonym'] = filled.loc[:, 'Pseudonym']
    for col in range(3, len(filled.columns)):
        spalte = filled.iloc[:, col]
        colname = filled.columns[col]
        if (spalte.isna().sum() >= blanknumber): 
            nonfilled[colname] = spalte
    colnamelist = list(nonfilled.columns)
    filled.drop(columns=colnamelist[1:], axis=1, inplace=True)
    return nonfilled, filled


def ABKUhaeufigkeit(names):
    df = pd.read_csv(str(names) + '.csv')
    dups_ABKU = df.pivot_table(columns=['ABKU'], aggfunc='size')
    dups_ABKU.to_csv('tmpdump.csv')
    dups_ABKU2 = pd.read_csv('tmpdump.csv')
    dups_ABKU2.rename(columns={'0': 'Anzahl'}, inplace=True)
    dups_ABKU2.sort_values('Anzahl', ascending=False, inplace=True)
    dups_ABKU2.to_csv(str(names) + 'ABKUHaeufigkeit.csv')
    return dups_ABKU2

            

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
    try:
        accuracy = (truePositiv + trueNegativ) / len(predicted)
    except:
        print('accuracy: Division by Zero')
    try:
        precision = truePositiv/(truePositiv+falsePositiv)
    except:
        print('precision: Division by Zero')
    try:
        recall = truePositiv/(truePositiv + falseNegativ)
    except:
        print('recall: Division by Zero')
    try:
        f1score = 2*(recall *precision)/(recall + precision)
    except:
        print('f1score: Division by Zero')
    
    return(accuracy, precision, recall, f1score)
   
       



def transp(inputDf):
    gesamt = pd.read_csv(str(inputDf), sep=';')
    completeDFcopy = gesamt.copy()
    completeDFcopy = gesamt.iloc[:, 3:]
    #Zeile löschen, die keine Untersuchungsart beinhaltet
    completeDFcopy['ABKU'].replace('', np.nan, inplace=True)
    completeDFcopy['Messwert_Zahl'].fillna( 0.0, inplace=True)
    completeDFcopy.dropna(subset=['ABKU'], inplace=True)
    completeDFcopy.sort_values(by=["Pseudonym", "relatives_datum", 'ABKU'], inplace=True, ascending=True)

    distinct_ABKU = completeDFcopy['ABKU'].unique()
    distinct_ABKUlist = distinct_ABKU.tolist()
    distinct_ABKUlist.append('Pseudonym')
    distinct_ABKUlist.append('relatives_datum')
    distinct_ABKU = np.array(distinct_ABKUlist)

    dictlist = [{}]
    datum = completeDFcopy.iloc[0, completeDFcopy.columns.get_loc('relatives_datum')]
    pseudo = completeDFcopy.iloc[0, completeDFcopy.columns.get_loc('Pseudonym')] #erstes Datum speichern
    tmpdict = {"Pseudonym" : str(pseudo), "relatives_datum" : str(datum)}
    counter = -1
    for index, row in completeDFcopy.iterrows():
        counter += 1
        if counter%100000 == 0:
            print('Zeile: ', counter, 'Pseudonym: ', row[4], 'Uhrzeit: ', time.strftime("%d.%m.%Y %H:%M:%S"))
        tmpdatum = row[3]#aktuelles Zeilen-Datum zwischenspeichern
        if tmpdatum != datum: 
            datum = tmpdatum
            dictlist.append(tmpdict)
            tmpdict = {}
            p = row[4]
            d = datum
            tmpdict['Pseudonym'] = str(p)
            tmpdict['relatives_datum'] = str(datum)
        
        tmpdict[str(row[2])] = str(row[1]) #ABKU:WERT(Messwert_Zahl)
        if (row[1] == np.nan) or (row[1]==0.0):
            tmpdict[str(row[2])] = row[0]  #Falls wert nan oder 0.0 ist, nimm wert aus Messwert_String

    
    dictlist.append(tmpdict)
    df = pd.DataFrame(dictlist)
    return df
        
            
def isfloat(value):
  try:
    float(value)
    return float(value)
  except ValueError:
    return np.nan

def addStatusModel1(addDF, statusDF):
    # Status hinzufügen
    tot = pd.read_csv(str(statusDF))
    preclassData = pd.read_csv(str(addDF))
    preclassData['Status'] = np.nan
    totenliste = tot.loc[:, 'Pseudonym'].tolist()
    for rows in range(len(preclassData)):
        tmpPseudo = preclassData.iloc[rows, preclassData.columns.get_loc('Pseudonym')]
        if totenliste.__contains__(tmpPseudo):
            preclassData.iloc[rows, preclassData.columns.get_loc('Status')] = 1 #1 = tot
        else:
            preclassData.iloc[rows, preclassData.columns.get_loc('Status')] = 0
    preclassData.to_csv(str(addDF))
    ############################################################################################


def takeLatest(inputDF, pseudo):
    dffilled = pd.read_csv(str(inputDF))
    dffilled = dffilled.iloc[:, 1:]
    p = dffilled[dffilled['Pseudonym'] == pseudo]
    p.sort_values(by='relatives_datum', ascending=False, inplace=True)
    col = dffilled.columns
    # tmp.to_csv('0.csv')
    naive = pd.DataFrame(columns=col)
    abkuset = set()
    for row in range(0, len(p)):
        for col in range(0, len(p.columns)):
            value = p.iloc[row, col]
            if (np.isnan(value) == False) and (abkuset.__contains__(p.columns[col]) == False): 
                naive.loc[0, p.columns[col]] = value
                abkuset.add(p.columns[col])
    
    return naive


def takeLatestAsDF(inputDF, pseudo):
    dffilled = inputDF.copy()
    # dffilled = dffilled.iloc[:, 1:]
    p = dffilled[dffilled['Pseudonym'] == pseudo]
    p.sort_values(by='relatives_datum', ascending=False, inplace=True)
    col = dffilled.columns
    # tmp.to_csv('0.csv')
    naive = pd.DataFrame(columns=col)
    abkuset = set()
    for row in range(0, len(p)):
        for col in range(0, len(p.columns)):
            value = p.iloc[row, col]
            if (np.isnan(value) == False) and (abkuset.__contains__(p.columns[col]) == False): 
                naive.loc[0, p.columns[col]] = value
                abkuset.add(p.columns[col])
    
    return naive
#####################################################################################################################################################
# # nehme Todesintervalle der toten Patienten als Klasse
# # Füge in Verstorben.csv eine Intervallspalte hinzu
def addtoverstorben(verstorbenDF, interval):
    verstorben = pd.read_csv(str(verstorbenDF))
    verstorben = verstorben.iloc[:, 1:]
    for row in range(len(verstorben)):
        tmp = 1
        todesdatum = verstorben.loc[row, 'relatives_datum']
        for el in interval:
            if (todesdatum >= el[0]) and (todesdatum <= el[1]):
                verstorben.loc[row, 'todesinterval'] = tmp
                break
            tmp += 1 
    return verstorben

def takelatestperInterval(nostrdf, interval):
    df = pd.read_csv(str(nostrdf))
    df = df.iloc[:, 1:]
    pseudo = df['Pseudonym'].unique()
    # tmp = df[(df['Pseudonym'] == 0) & (df['relatives_datum'] >= intervalle[0][0]) & (df['relatives_datum'] <= intervalle[0][1])]
    # print(tmp)
    frames = []
    for name in pseudo:
        for el in interval:
            tmpIntervalDF = df[(df['Pseudonym'] == name) & (df['relatives_datum'] >= el[0]) & (df['relatives_datum'] <= el[1])]
            tmpdf = takeLatestAsDF(tmpIntervalDF, name)
            frames.append(tmpdf)
    result = pd.concat(frames)
    return result

def fillmodeperPseudo(latestdf):
    df = pd.read_csv(str(latestdf))
    df = df.iloc[:, 1:]
    pseudo = df['Pseudonym'].unique()
    frames = []
    for name in pseudo:
        tmpdf = df.loc[df['Pseudonym'] == name]
        for i in tmpdf.columns[tmpdf.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
            ser = tmpdf.loc[:, i]
            tmpdf[i].fillna(ser.mode(),inplace=True)
        frames.append(tmpdf)
    result = pd.concat(frames)
    return result

def fillintervalstatus(classificationabletable, interval):
    df = pd.read_csv(str(classificationabletable))
    statusDF = pd.read_csv('Verstorben_Interval.csv')
    statusDF = statusDF.iloc[:, 1:]
    status = statusDF['Pseudonym'].unique()
    df = df.iloc[:, 1:]
    df['status'] = len(interval) + 1
    index = df.index
    pseudo = df['Pseudonym'].unique()
    for row in range(len(df)):
        name = df.loc[row, 'Pseudonym']
        if status.__contains__(name):
            todesint = statusDF.query('Pseudonym == ' + str(name))['todesinterval'].to_list()
            df.loc[row, 'status'] = todesint[0]
    return df

