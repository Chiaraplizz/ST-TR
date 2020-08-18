import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
data_baseline = np.load('/Users/chiaraplizzari/Desktop/ECCV_code/confusion_baseline.npy')
data_spatial = np.load('/Users/chiaraplizzari/Desktop/ECCV_code/confusion_temporal_cat.npy')
#Calculates and plots the confusion matrix


print(data_baseline)
print(data_spatial)

df_cm = pd.DataFrame(data_baseline,
                         index=[i for i in range(1, 61)],
                         columns=[i for i in range(1, 61)], dtype=np.int)
print(df_cm)
plt.figure(figsize=(30, 24))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.savefig('/Users/chiaraplizzari/Desktop/confusion_matrices/data_baseline.png')

df_cm1 = pd.DataFrame(data_spatial,
                         index=[i for i in range(1, 61)],
                         columns=[i for i in range(1, 61)], dtype=np.int)
print(df_cm1)
plt.figure(figsize=(30, 24))
sn.heatmap(df_cm1, annot=True, cmap='Blues', fmt='g')
plt.savefig('/Users/chiaraplizzari/Desktop/confusion_matrices/data_temp_cat.png')
plt.show()