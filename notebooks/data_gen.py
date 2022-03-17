# Load Libraries

import numpy as np
import scipy.io
import glob

from preproc import bp_filter, epochs_gen

# Set Seed
np.random.seed(13)

# Importing the dataset
filepath = "/workspace/data/EEG/data/BNCI/"
df = glob.glob(filepath + '*.mat')

#TODO: Make a dict
df1 = scipy.io.loadmat(df[0])
df2 = scipy.io.loadmat(df[1])
df3 = scipy.io.loadmat(df[2])
df4 = scipy.io.loadmat(df[3])
df5 = scipy.io.loadmat(df[4])
df6 = scipy.io.loadmat(df[5])
df7 = scipy.io.loadmat(df[6])
df8 = scipy.io.loadmat(df[7])

jdf = [df1, df2, df3, df4, df5, df6, df7, df8]

del df1, df2, df3, df4, df5, df6, df7, df8, df

#TODO: Make a dict
filt_raw = []
ep = []
ev = []
for i in range(len(jdf)):
    filt_raw.append(bp_filter(jdf[i]))
    ep.append(epochs_gen(filt_raw[i]))
    ev.append(ep[i].average(by_event_type=True))

for i in range(len(ep)):
    ep[i].save('/workspace/data/EEG/data/epochs/A0' + str(i + 1) + '-epo.fif')

#Label extraction
l1 = ep[0].events[:,-1]
l2 = ep[1].events[:,-1]
l3 = ep[2].events[:,-1]
l4 = ep[3].events[:,-1]
l5 = ep[4].events[:,-1]
l6 = ep[5].events[:,-1]
l7 = ep[6].events[:,-1]
l8 = ep[7].events[:,-1]

X1 = ep[0].get_data()
X2 = ep[1].get_data()
X3 = ep[2].get_data()
X4 = ep[3].get_data()
X5 = ep[4].get_data()
X6 = ep[5].get_data()
X7 = ep[6].get_data()
X8 = ep[7].get_data()

#Combine epochs and labels
combined_epochs = np.stack((X8, X7, X6, X5, X4, X3, X2, X1), axis=0)
combined_labels = np.stack((l1,l2,l3,l4,l5,l6,l7,l8), axis=0)
combined_labels = combined_labels-1

#Write as arrays
np.save('/workspace/data/EEG/data/epochs/A-epo.npy', combined_epochs)
np.save('/workspace/data/EEG/data/epochs/A-labels.npy', combined_labels)