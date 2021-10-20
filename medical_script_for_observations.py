from matplotlib import pyplot as plt
import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

onlyNonNumbers = pd.read_csv('dfNonNumberOnly.csv')
onlyNumbers = pd.read_csv('dfNumbersOnly.csv')

alb = onlyNonNumbers['ALB']
try:
    for el in range(len(alb)):
	    alb[el] = alb[el].astype(float)
except:
	alb[el] = np.nan

print(alb)
