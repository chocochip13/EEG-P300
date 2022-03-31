# Load Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import *
from utils import *
import tensorflow.keras.backend as K

import time

from plot_aucs import plot_cross_aucs
from model_sepconv1d import SepConv1D

# Load Data
CE = np.load('/workspace/data/EEG/data/epochs/A-epo.npy')
lab = np.load("/workspace/data/EEG/data/epochs/A-labels.npy")

tCE = np.transpose(CE, (0, 1, 3, 2))  # Reshape as per model requirement


def evaluate_cross_subject_model(data, labels, modelpath, n_filters = 32):
    """
    Trains and evaluates SepConv1D for each subject in the P300 Speller database
    using random cross validation.
    """
    n_sub = data.shape[0]
    n_ex_sub = data.shape[1]
    n_samples = data.shape[2]
    n_channels = data.shape[3]

    aucs = np.zeros(n_sub)
    inf_time = np.zeros(n_sub)

    data = data.reshape((n_sub * n_ex_sub, n_samples, n_channels))
    labels = labels.reshape((n_sub * n_ex_sub))
    groups = np.array([i for i in range(n_sub) for j in range(n_ex_sub)])

    cv = LeaveOneGroupOut()
    for k, (t, v) in enumerate(cv.split(data, labels, groups)):
        X_train, y_train, X_test, y_test = data[t], labels[t], data[v], labels[v]
        rg = np.random.choice(t, 1)
        sv = groups[t] == groups[rg]
        st = np.logical_not(sv)
        X_train, y_train, X_valid, y_valid = data[t][st], labels[t][st], data[t][sv], labels[t][sv]
        print("Partition {0}: train = {1}, valid = {2}, test = {3}".format(k, X_train.shape, X_valid.shape, X_test.shape))
        print("Groups train = {0}, valid = {1}, test = {2}".format(np.unique(groups[t][st]),
                                                                   np.unique(groups[t][sv]),
                                                                   np.unique(groups[v])))

        # channel-wise feature standarization
        sc = EEGChannelScaler(n_channels = n_channels)
        X_train = sc.fit_transform(X_train)
        X_valid = sc.transform(X_valid)
        X_test = sc.transform(X_test)

        model = SepConv1D(Chans = n_channels, Samples = n_samples, Filters = n_filters)
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

        es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, restore_best_weights = True)

        start_train = time.time()
        model.fit(X_train,
                  y_train,
                  batch_size = 256,
                  epochs = 200,
                  validation_data = (X_valid, y_valid),
                  callbacks = [es])
        train_time = time.time()-start_train
        print(train_time)

        start_test = time.time()
        proba_test = model.predict(X_test)
        test_time = time.time() - start_test

        test_size = X_test.shape[0]
        inf_time[k] = test_time/test_size

        aucs[k] = roc_auc_score(y_test, proba_test)
        print('P{0} -- AUC: {1}'.format(k, aucs[k]))
        model.save_weights(modelpath + '/s' + str(np.unique(groups[v])[0]) + '_model.h5')
        np.save(modelpath + '/s' + str(np.unique(groups[v])[0]) + '_data.npy', X_test)
        np.save(modelpath + '/s' + str(np.unique(groups[v])[0]) + '_labels.npy', y_test)
        K.clear_session()

    np.savetxt(modelpath + '/aucs.npy', aucs)
    np.savetxt(modelpath + '/inf_time.npy', inf_time)


cross_modelpath = "/workspace/data/EEG/models/sepconv1d/cross/"
evaluate_cross_subject_model(tCE, lab, cross_modelpath, n_filters = 32)

plot_cross_aucs(cross_modelpath, cross_modelpath+"sepconv1d_cross_auc.png", "SepConv1d - Cross")
