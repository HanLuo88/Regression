import unittest
import pandas as pd


class TestStringMethods(unittest.TestCase):

    def test_feature_by_Distance(self):
        automatic = True
        automaticDistance = 0.3
        inputFile = 'naive_latest_todesinterval_model3.csv'
        distanceFile = 'M4_means_Distance.csv'
        outputFile = 'model4.csv'

        df = pd.read_csv(inputFile)
        df = df.iloc[:, 1:]

        cols = ['Pseudonym', 'relatives_datum', 'CRP', 'GGT37', 'GOT37', 'GPT37', 'HB', 'LEUKO', 'THROMB', 'LDH37',
                'M-BL', 'FERR', 'BK-PCR', 'CMV-DNA', 'EBV-DNA', 'HST', 'M-AZ', 'M-NRBC', 'M-PR', 'NRBC-ABS', 'IL-6', 'status']

        input = pd.read_csv(distanceFile)
        tmp = input[input['relative_Abweichung'] >= automaticDistance]
        tmpcols = ['Pseudonym', 'relatives_datum']
        for el in tmp['Unnamed: 0.1']:
            print(el)
            tmpcols.append(el)
        tmpcols.append('status')

        if automatic:
            cols = tmpcols

        frames = []
        for el in cols:
            tmpseries = df.loc[:, el]
            frames.append(tmpseries)
        result = pd.concat(frames, axis=1)
        result.to_csv(outputFile)

    def test_remove_verificationPatients(self):
        inputFile = 'model4.csv'
        outputFile = 'model4_TMP.csv'

        toRemoveSurvivor = [124, 3297, 6658, 282441]
        toRemoveDead = [1235, 3487, 5865, 8730]

        df = pd.read_csv(inputFile)

        tmpDf = df.iloc[:, 1:]
        for index in range(0, len(toRemoveSurvivor)):

            tot_M4 = df[df['Pseudonym'] == toRemoveDead[index]]
            tot_M4.drop('status', inplace=True, axis=1)
            tot_M4.to_csv('p' + str(toRemoveDead[index]) + '_M4.csv')

            lebend_M4 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
            lebend_M4.drop('status', inplace=True, axis=1)
            lebend_M4.to_csv('p' + str(toRemoveSurvivor[index]) + '_M4.csv')
            
            tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

        tmpDf.to_csv(outputFile)


if __name__ == '__main__':
    unittest.main()
