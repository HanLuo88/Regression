import unittest
import pandas as pd
import medical_lib as ml

class TestStringMethods(unittest.TestCase):

    def test_Zeitraum_Eingrenzung(self):
        intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
        # Eingrenzung des Betrachtungszeitraumes:
        # Model 2 teilt den Zeitraum in intervalle und nimmt pro Patient den letzten Wert in diesem Intervall pro Zeile.
        # Dadurch hat jeder Patient mehr als eine Zeile am ende.
        # Nach Tag 0 anhand der Verteilung der relativen Tage:  Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü + 3S = 2031.95 Tage = 2032 Tage
        # Vor Tag 0 anhand der Verteilung der relativen Tage: Mittelwert mü = 118.58 Tage, SIgma s = 637,79 Tage => mü - s = -519.21 Tage = -520 Tage
        dfcomplete = pd.read_csv('transposed_complete.csv')
        dfcomplete = dfcomplete.iloc[:, 1:]
        dfcomplete = dfcomplete[(dfcomplete.relatives_datum < 1620)
                                & (dfcomplete.relatives_datum > -520)]
        dfcomplete.reset_index(drop=True, inplace=True)
        dfcomplete.to_csv('transposed_complete_model1.csv')
        # Es sind nur etwa 10.000 Zeilen weniger als vorher, also ist diese Eingrenzung vertretbar.
############################################################################################################################
    def test_Patienten_loeschen_nach_1620_Tagen(self):
        # #löschen der vier Patienten, die nach 1620Tagen gestorben sind
        df = pd.read_csv('transposed_complete_model1.csv')
        df = df.iloc[:, 1:]
        df2 = df[(df['Pseudonym'] != 11952) & (df['Pseudonym'] != 86166) & (df['Pseudonym'] != 90954) & (df['Pseudonym'] != 231731)]
        df2.to_csv('transposed_complete_model1.csv')
############################################################################################################################    
    def test_Nicht_Gut_Gefuellte_Features_Loeschen(self):
        blanks = 90000
        notwellFilledDF = 'notFilled.csv'
        wellFilledDf = 'filled.csv'
        wellFilledNoStrings = 'filled_noStr.csv'


        #Entfernt alle nicht gut gefuellten Spalten
        dfNonFilled, dfFilled = ml.removeColsNotWellFilled('transposed_complete_model1.csv', blanks)
        dfNonFilled.to_csv(notwellFilledDF)
        # print('Fertig')
        dfFilled.to_csv(wellFilledDf)
        # print('Fertig')
        ###########################################################################################################################
        dffilled1 = pd.read_csv(wellFilledDf)
        dffilled1 = dffilled1.iloc[:, 1:]
        # print(type(dffilled1.loc[0, 'XDSON']))
        #Ersetzt 'neg' mit einer 0
        dffilled1.replace("neg", 0.0, inplace=True)
        #Prueft ob ein alphanumerischer Eintrag in ein Float umgewandelt werden kann. Falls nicht, wird es mit einem np.nan ersetzt
        dffilled1 = dffilled1.applymap(func=ml.isfloat)
        #Entfernt Spalten, die komplett aus nans Bestehen
        dffilled1.dropna(axis=1, how='all', inplace=True)
        # print(dffilled1['XDSON'].dtypes)
        dffilled1.to_csv(wellFilledNoStrings)
        ###########################################################################################################################
        dffilled2 = pd.read_csv(wellFilledNoStrings)
        dffilled2 = dffilled2.iloc[:, 1:]
        dffilled2.drop('XDSON', inplace=True, axis=1)
        dffilled2.to_csv(wellFilledNoStrings)
############################################################################################################################
    def test_Aktuellste_Werte(self):
        #Erstelle Dataframe, in der für jeden Patienten nur sein aktuellster Wert pro Feature benutzt wird
        inputDf = 'filled_noStr.csv'
        outputDf = 'naive_latest.csv'
        

        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        pseudolist = df['Pseudonym'].unique()
        frames = []
        i = 0
        for name in pseudolist:
            tmp = ml.takeLatest(inputDf, name)
            frames.append(tmp)
            i += 1
        result = pd.concat(frames)
        result.to_csv(outputDf)
############################################################################################################################
    def test_Status_setzen(self):
        inputDf = 'naive_latest.csv'
        statusDf = 'Verstorben.csv'

        #Fügt Status zur InputDf hinzu
        ml.addStatusModel1(inputDf, statusDf)

        #Füllt Spaltenweise NaNs mit dem Spaltenmittelwert
        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
            df[i].fillna(df[i].mean(),inplace=True)
        df.to_csv(inputDf)                                #Schreibt das Endergebnis wieder in inputDf
############################################################################################################################
    def test_remove_verificationPatients(self):
        inputFile = 'naive_latest.csv'
       
        outputFile = 'naive_latest_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)
        df = df.iloc[:, 1:]

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M1 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M1.drop('Status', inplace=True, axis=1)
            tot_M1.to_csv('p' + str(toRemoveDead[index]) + '_M1.csv')

            lebend_M1 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M1.drop('Status', inplace=True, axis=1)
            lebend_M1.to_csv('p' + str(toRemoveSurvivor[index]) + '_M1.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)



if __name__ == '__main__':
    unittest.main()