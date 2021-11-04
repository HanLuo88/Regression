import unittest
import pandas as pd
import medical_lib as ml

class TestStringMethods(unittest.TestCase):
    def test_set_Interval(self):
        #Setzt Interval und überträgt es in die verstorben liste
        intervalle = [(-520, -200),(-199, 0),(1, 120),(121, 300),(301, 800),(801, 1650)]
        statusdf = 'Verstorben.csv'

        totintervalle = ml.addtoverstorben(statusdf, intervalle)
        totintervalle.to_csv('Verstorben_Interval.csv')
    


    def test_todesinterval_einfuegen(self):
        #Dataframe einlesen
        intervalle = [(-520, -200),(-199, 0),(1, 120),(121, 300),(301, 800),(801, 1650)]
        inputDf = 'naive_latest.csv'
        outputDf = 'naive_latest_M3.csv'
        todesintervaloutputDf = 'naive_latest_todesinterval_model3.csv'


        df = pd.read_csv(inputDf)
        df = df.iloc[:, 2:]
        df.drop('Status', inplace=True, axis=1)
        df.to_csv(outputDf)
        #Todesinterval einfügen
        df = ml.fillintervalstatus(outputDf, intervalle)
        df.to_csv(todesintervaloutputDf)

    def test_remove_verificationPatients(self):
        inputFile = 'naive_latest_todesinterval_model3.csv'
       
        outputFile = 'naive_latest_model3_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)
        # df = df.iloc[:, 1:]

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M3 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M3.drop('status', inplace=True, axis=1)
            tot_M3.to_csv('p' + str(toRemoveDead[index]) + '_M3.csv')

            lebend_M3 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M3.drop('status', inplace=True, axis=1)
            lebend_M3.to_csv('p' + str(toRemoveSurvivor[index]) + '_M3.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)




if __name__ == '__main__':
    unittest.main()
