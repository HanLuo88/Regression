from matplotlib import colors, pyplot as plt
from pandas.core.algorithms import mode
from pandas.io.parsers import read_csv
import medical_lib as ml
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import linregress

df = pd.read_csv('automated_algorithmen.csv')
df = df.iloc[:, 1:]
models = df.iloc[:, 0]
knn = df.iloc[:, 1].to_numpy()
logreg =  df.iloc[:, 2].to_numpy()
dectree = df.iloc[:, 3].to_numpy()
rf = df.iloc[:, 4].to_numpy()
ada = df.iloc[:, 5].to_numpy()
xgm = df.iloc[:, 6].to_numpy()

# plt.rcParams['figure.figsize'] = [35,10]
# plt.plot(models,knn,  label = "KNN", linestyle='-', marker = 'o', linewidth = 2.5, color = 'black')
# plt.bar(models,knn, width=0.01)
# plt.plot(models,logreg,  label = "Logistic Regression", linestyle='-', marker = 'o', linewidth = 2.5, color = 'lime')
# plt.bar(models,logreg, width=0.01)
# plt.plot(models,dectree,  label = "Decision Tree", linestyle='-', marker = 'o', linewidth = 2.5, color = 'red')
# plt.bar(models,dectree,  width=0.01)
# plt.plot(models,rf,  label = "Random Forest", linestyle='-', marker = 'o', linewidth = 2.5, color = 'saddlebrown')
# plt.bar(models,rf, width=0.01)
# plt.plot(models,ada,  label = "ADABoost", linestyle='-', marker = 'o', linewidth = 2.5, color = 'darkviolet')
# plt.bar(models,ada,  width=0.01)
# plt.plot(models,xgm,  label = "XGBoost", linestyle='-', marker = 'o', linewidth = 2.5, color = 'violet')
# plt.bar(models,xgm,  width=0.01)
# plt.legend()
# plt.show()

########################################################################################################################################################
# knn_normal = [knn[0], knn[1], knn[4], knn[5], knn[8], knn[9], knn[12], knn[13]]
# logreg_normal = [logreg[0], logreg[1], logreg[4], logreg[5], logreg[8], logreg[9], logreg[12], logreg[13]]
# dectree_normal_normal = [dectree[0], dectree[1], dectree[4], dectree[5], dectree[8], dectree[9], dectree[12], dectree[13]]
# rf_normal = [rf[0], rf[1], rf[4], rf[5], rf[8], rf[9], rf[12], rf[13]]
# ada_normal = [ada[0], ada[1], ada[4], ada[5], ada[8], ada[9], ada[12], ada[13]]
# xgm_normal = [xgm[0], xgm[1], xgm[4], xgm[5], xgm[8], xgm[9], xgm[12], xgm[13]]

# labels = ['Model 1', 'Model 1 Selected', 'Model 2', 'Model 2 Selected', 'Model 3', 'Model 3 Selected', 'Model 4', 'Model 4s Selected']
# plt.rcParams['figure.figsize'] = [15,15]
# x = np.arange(len(labels))  # the label locations
# width = 0.1  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x, knn_normal, width, label='knn')
# rects2 = ax.bar(x + 0.1, logreg_normal, width, label='logreg')
# rects3 = plt.bar(x + 0.2, dectree_normal_normal, width, label='dectree')
# rects4 = plt.bar(x + 0.3, rf_normal, width, label='rf')
# rects5 = plt.bar(x + 0.4, ada_normal, width, label='ada')
# rects6 = plt.bar(x + 0.5, xgm_normal, width, label='xgm')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Precision')
# ax.set_title('Grouped by Model')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()


# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)
# autolabel(rects4)
# autolabel(rects5)
# autolabel(rects6)
# # fig.tight_layout()

# plt.show()




knn_v2 = [knn[2], knn[3], knn[6], knn[7], knn[10], knn[11], knn[14], knn[15]]
logreg_v2 =[logreg[2], logreg[3], logreg[6], logreg[7], logreg[10], logreg[11], logreg[14], logreg[15]]
dectree_v2 = [dectree[2], dectree[3], dectree[6], dectree[7], dectree[10], dectree[11], dectree[14], dectree[15]]
rf_v2 = [rf[2], rf[3], rf[6], rf[7], rf[10], rf[11], rf[14], rf[15]]
ada_v2 = [ada[2], ada[3], ada[6], ada[7], ada[10], ada[11], ada[14], ada[15]]
xgm_v2 = [xgm[2], xgm[3], xgm[6], xgm[7], xgm[10], xgm[11], xgm[14], xgm[15]]

labels = ['Model 1 v2', 'Model 1 Selected v2', 'Model 2 v2', 'Model 2 Selected v2', 'Model 3 v2', 'Model 3 Selected v2', 'Model 4 v2', 'Model 4s Selected v2']
plt.rcParams['figure.figsize'] = [15,15]
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, knn_v2, width, label='knn')
rects2 = ax.bar(x + 0.1, logreg_v2, width, label='logreg')
rects3 = plt.bar(x + 0.2, dectree_v2, width, label='dectree')
rects4 = plt.bar(x + 0.3, rf_v2, width, label='rf')
rects5 = plt.bar(x + 0.4, ada_v2, width, label='ada')
rects6 = plt.bar(x + 0.5, xgm_v2, width, label='xgm')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision')
ax.set_title('Grouped by Model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)
# fig.tight_layout()

plt.show()