import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


N = 60
K = 1
bins = 200
df = pd.read_csv("DATA/data_img_control.csv", header=None)
n, bins, patches =  plt.hist(df[1], bins=bins)
print(df[1].value_counts())
angles = np.array(df[1])
n = np.array(n)

idx = n.argsort()[-K:][::-1]    # find the largest K bins
del_ind = []                    # collect the index which will be removed from the data
for i in range(K):
    if n[idx[i]] > N:
        ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
        ind = np.ravel(ind)
        np.random.shuffle(ind)
        del_ind.extend(ind[:len(ind)-N])

# angles = np.delete(angles,del_ind)
balanced_samples = [v for i, v in enumerate(df) if i not in del_ind]
# print(balanced_samples)
balanced_angles = np.delete(angles,del_ind)

plt.figure()
plt.hist(balanced_angles, bins=bins, color= 'orange', linewidth=0.1)
plt.title('modified histogram', fontsize=20)
plt.xlabel('steering angle', fontsize=20)
plt.ylabel('counts', fontsize=20)
plt.show()