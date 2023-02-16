import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


N = 60
K = 1
bins = 200
data_list = glob.glob('DATA'+'/*/data.csv')
data_df = pd.read_csv(data_list[0], header = None)
for data in data_list[1:]:
    df = pd.read_csv(data, header = None)
    # print(df)
    data_df = data_df.append(df, ignore_index = True)
n, bins, patches =  plt.hist(data_df[1], bins=bins)

print(data_df)
print(data_df[1].value_counts())
angles = np.array(data_df[1])
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