from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

beides = [468821.0, 546239.0, 433748.0, 123038.0, 437426.0, 384002.0, 152315.0, 2832.0, 193588.0, 305935.0, 218655.0, 479749.0, 425179.0, 214386.0, 112457.0, 65046.0, 1256.0, 347528.0, 58330.0, 90921.0, 297516.0, 276504.0, 519749.0, 436218.0, 105083.0, 1938.0, 286620.0, 223203.0, 381014.0, 103967.0, 145128.0, 137176.0, 358833.0, 185008.0, 137406.0, 229869.0, 159184.0, 6658.0, 487090.0, 109351.0, 85038.0, 153320.0, 151085.0, 352152.0, 437538.0, 317660.0, 158130.0, 501778.0, 261102.0, 143683.0, 206465.0, 250330.0, 150719.0, 148676.0, 512292.0, 271693.0, 455282.0, 74731.0, 53045.0, 3297.0, 399442.0, 11288.0, 522905.0, 557647.0, 100030.0, 11862.0, 52103.0, 277820.0, 109365.0, 200428.0, 136616.0, 143873.0, 216569.0, 510367.0, 463977.0, 420434.0, 239184.0, 146025.0, 4747.0, 344065.0, 253091.0, 368302.0, 39361.0, 2628.0, 339886.0, 5650.0, 249453.0, 79.0, 351602.0, 248969.0, 55973.0, 223584.0, 522202.0, 314110.0, 70129.0, 235689.0, 268835.0, 21470.0, 343394.0, 270339.0,
          308266.0, 400060.0, 216537.0, 18298.0, 428551.0, 80660.0, 308726.0, 254265.0, 502516.0, 6360.0, 359675.0, 301913.0, 124, 282441, 0, 247, 1068, 1090, 1235, 1414, 2848, 3487, 4811, 5865, 7233, 8440, 8730, 9828, 10037, 11849, 18425, 22278, 23649, 34038, 43249, 46549, 50183, 53835, 55997, 65201, 66153, 70862, 71242, 75011, 75756, 88775, 90326, 90836, 91102, 97831, 97875, 97896, 102395, 108556, 108560, 115135, 117122, 121883, 121992, 126150, 129784, 131293, 132559, 138631, 139733, 141378, 145421, 149121, 153152, 160758, 164028, 164720, 164722, 165637, 170102, 172954, 179563, 180192, 182214, 191305, 197629, 199491, 199768, 207620, 213765, 218249, 218700, 222605, 230667, 231763, 249985, 255576, 264955, 265696, 276359, 278289, 280462, 280549, 289322, 295113, 296988, 297726, 299227, 302096, 305091, 307036, 311805, 313781, 316005, 319422, 320168, 328592, 329269, 335889, 349211, 355925, 358574, 383170, 390327, 405077, 433546, 438702, 449362, 453517, 465292, 471085, 475753, 511416]

df = pd.read_csv('naive_latest_todesinterval_model3.csv')
df = df.iloc[:, 1:]
df1 = df[df['Pseudonym'].isin(beides)]
df1.to_csv('naive_latest_todesinterval_model3_v2.csv')
################################################################################################################################################
# inputFile = 'naive_latest_todesinterval_model3_v2.csv'
       
# outputFile = 'naive_latest_todesinterval_model3_v2_TMP.csv'

# toRemoveSurvivor = [124, 3297, 6658, 282441]
# toRemoveDead = [1235, 3487, 5865, 8730]

# df = pd.read_csv(inputFile)

# tmpDf = df.iloc[:, 1:]
# for index in range(0, len(toRemoveSurvivor)):
#     tot_M1 = df[df['Pseudonym'] == toRemoveDead[index]]
#     tot_M1.drop('status', inplace=True, axis=1)
#     tot_M1.to_csv('p' + str(toRemoveDead[index]) + '_M3_v2.csv')

#     lebend_M1 = df[df['Pseudonym'] == toRemoveSurvivor[index]]
#     lebend_M1.drop('status', inplace=True, axis=1)
#     lebend_M1.to_csv('p' + str(toRemoveSurvivor[index]) + '_M3_v2.csv')
            
#     tmpDf = tmpDf[(df['Pseudonym'] != toRemoveSurvivor[index]) & (df['Pseudonym'] != toRemoveDead[index])]

# tmpDf.to_csv(outputFile)