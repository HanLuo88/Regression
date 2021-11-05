import unittest
import pandas as pd
import medical_lib as ml


class TestStringMethods(unittest.TestCase):

    def test_Zeitraum_Eingrenzung(self):
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
        dfcomplete.to_csv('transposed_model2.csv')
        # Es sind nur etwa 10.000 Zeilen weniger als vorher, also ist diese Eingrenzung vertretbar.
############################################################################################################################
    def test_Patienten_loeschen_nach_1620_Tagen(self):
        # #löschen der vier Patienten, die nach 1620Tagen gestorben sind
        df = pd.read_csv('transposed_model2.csv')
        df = df.iloc[:, 1:]
        df2 = df[(df['Pseudonym'] != 11952) & (df['Pseudonym'] != 86166) & (df['Pseudonym'] != 90954) & (df['Pseudonym'] != 231731)]
        df2.to_csv('transposed_model2.csv')
############################################################################################################################
    def test_set_Interval(self):
        #Setzt Interval und überträgt es in die verstorben liste
        intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
        statusdf = 'Verstorben.csv'

        totintervalle = ml.addtoverstorben(statusdf, intervalle)
        totintervalle.to_csv('Verstorben_Interval.csv')
############################################################################################################################    
    def test_prepareDataframe(self):
        #Vorbereiten von transposed:model2.csv
        inputDf = 'transposed_model2.csv'
        notwellFilledDF = 'model2notFilled.csv'
        wellFilledDf = 'model2filled.csv'
        wellFilledNoStrings = 'model2filled_noStr.csv'
        blanks = 90000

        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        model2nonfilled, model2filled = ml.removeColsNotWellFilled(inputDf, blanks)
        model2nonfilled.to_csv(notwellFilledDF)
        model2filled.to_csv(wellFilledDf)
        ############################################################################################################################
        model2filled1 = pd.read_csv(wellFilledDf)
        model2filled1 = model2filled1.iloc[:, 1:]

        model2filled1.replace("neg", 0.0, inplace=True)
        model2filled1 = model2filled1.applymap(func=ml.isfloat)
        model2filled1.dropna(axis=1, how='all', inplace=True)

        model2filled1.to_csv(wellFilledNoStrings)
        ###########################################################################################################################
        model2filled2 = pd.read_csv(wellFilledNoStrings)
        model2filled2 = model2filled2.iloc[:, 1:]
        tmpPseudo = model2filled2.iloc[:, 0]
        tmpDatum = model2filled2.iloc[:, 1]
        df1 = model2filled2.iloc[:, 2:]

        df1 = df1.dropna(how='all')
        df1.insert(loc=0, column='relatives_datum', value=tmpDatum)

        df1.insert(loc=0, column='Pseudonym', value=tmpPseudo)

        df1.drop('XDSON', inplace=True, axis=1)
        df1.to_csv(wellFilledNoStrings)
############################################################################################################################
    def test_Aktuellste_Werte_Pro_Interval(self):
        #Pro Intervall den neusten Wert des Intervalls nehmen
        intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
        inputdf = 'model2filled_noStr.csv'
        outputdf = 'model2_interval_latest.csv'

        df = ml.takelatestperInterval(inputdf, intervalle)
        df.to_csv(outputdf)
############################################################################################################################
    def test_fillmean_pro_interval(self):
        # Pro Patient werden leere Zellen werden mit dem mean des Intervalls gefüllt
        inputDf = 'model2_interval_latest.csv'
        outputDf = 'model2_Classificationtable_intervalstatus.csv'

        df = ml.fillmeanperPseudo(inputDf)
        df.to_csv(outputDf)
############################################################################################################################
    def test_fill_rest(self):
        # # Pro Patient werden leere Zellen werden mit dem mean des Intervals gefüllt
        inputDf = 'model2_Classificationtable_intervalstatus.csv'
        df = pd.read_csv(inputDf)
        df = df.iloc[:, 1:]
        for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
                df[i].fillna(df[i].mean(),inplace=True)
        df.to_csv(inputDf)
############################################################################################################################
    def test_set_IntervalStatus(self):
        intervalle = [(-520, -200),(-199, 0),(1, 14),(15, 30),(31, 60),(61,90),(91,120),(121,180),(181,365),(366,850),(851,1650)]
        inputDf = 'model2_Classificationtable_intervalstatus.csv'

        df = ml.fillintervalstatus(inputDf, intervalle)
        df.to_csv(inputDf)
############################################################################################################################
    def test_remove_verificationPatients(self):
        inputFile = 'model2_Classificationtable_intervalstatus.csv'
       
        outputFile = 'model2_Classificationtable_intervalstatus_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M2 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M2.drop('status', inplace=True, axis=1)
            tot_M2.to_csv('p' + str(toRemoveDead[index]) + '_M2.csv')

            lebend_M2 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M2.drop('status', inplace=True, axis=1)
            lebend_M2.to_csv('p' + str(toRemoveSurvivor[index]) + '_M2.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)



if __name__ == '__main__':
    unittest.main()
