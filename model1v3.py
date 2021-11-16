from enum import unique
from scipy.sparse.construct import rand
from sklearn.utils import shuffle
import medical_lib as ml
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

#Model1v2 ist model1, jedoch gibt es genauso viele Überlebende wie Tote und getestet wir mit unbalancierten Dataset. Die Werte sind jeweils die direkt nach der OP
df = pd.read_csv('filled_noStr.csv')
df = df.iloc[:, 1:]

tot = pd.read_csv('Verstorben.csv')

uniquetote = tot['Pseudonym'].unique().tolist()

df1 = df[df['Pseudonym'].isin(uniquetote)] #Dataframe nur mit den Toten
df2 = df[~df['Pseudonym'].isin(uniquetote)] #Dataframe nur mit den Lebenden
uniquelebende = df2['Pseudonym'].unique().tolist() #Liste aller distinct Lebende

uniquetote1 = uniquetote[0:(len(uniquetote)-20)] #Liste aus 94 tote
uniquetote2 = uniquetote[len(uniquetote)-20: len(uniquetote)] #Liste der restlichen 20tote

uniquelebende1 = uniquelebende[0: 94] #Liste aus 94Lebende
uniquelebende2 = uniquelebende[94:len(uniquelebende)] # Liste der restlichen Lebenden

beides_Train = uniquetote1 + uniquelebende1
beides_Test = uniquetote1 + uniquelebende2

df1v3_train = df[df['Pseudonym'].isin(beides_Train)] # Dataframe aus 94tote und 94lebende. Komplettes Train-Set
df1v3_Test = df[df['Pseudonym'].isin(beides_Test)]
df1v3_train.to_csv('model1v3trainunfilled.csv')
df1v3_Test.to_csv('model1v3testunfilled.csv')
