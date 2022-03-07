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

filt_raw = []
ep = []
ev = []
for i in range(len(jdf)):
    filt_raw.append(bp_filter(jdf[i]))
    ep.append(epochs_gen(filt_raw[i]))
    ev.append(ep[i].average(by_event_type=True))

for i in range(len(ep)):
    ep[i].save('data/epochs/epoch-epo-A0' + str(i) + '.fif')
