# Load Libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import *
from utils import *
import tensorflow.keras.backend as K

import time

from plot_aucs import plot_within_aucs
from model_sepconv1d import SepConv1D

# Load Data
CE = np.load('/workspace/data/EEG/data/epochs/A-epo.npy')
lab = np.load("/workspace/data/EEG/data/epochs/A-labels.npy")

tCE = np.transpose(CE, (0, 1, 3, 2))  # Reshape as per model requirement

def evaluate_subject_models(data, labels, modelpath, subject, n_filters = 16):
    """
    Trains and evaluates P300-CNNT for each subject in the P300 Speller database
    using repeated stratified K-fold cross validation.
    """
    n_sub = data.shape[0]
    n_trials = data.shape[1]
    n_samples = data.shape[2]
    n_channels = data.shape[3]

    inf_time = np.zeros(5 * 10)
    aucs = np.zeros(5 * 10)

    print("Training for subject {0}: ".format(subject))
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state = 123)
    for k, (t, v) in enumerate(cv.split(data[subject], labels[subject])):
        X_train, y_train, X_test, y_test = data[subject, t, :, :], labels[subject, t], data[subject, v, :, :], labels[subject, v]
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, shuffle = True, random_state = 456)
        print('Partition {0}: X_train = {1}, X_valid = {2}, X_test = {3}'.format(k, X_train.shape, X_valid.shape, X_test.shape))

        # channel-wise feature standarization
        sc = EEGChannelScaler(n_channels = n_channels)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = SepConv1D(Chans = n_channels, Samples = n_samples, Filters = n_filters)
        print(model.summary())
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

        # Early stopping setting also follows EEGNet (Lawhern et al., 2018)
        es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 50, restore_best_weights = True)

        start_train = time.time()
        history = model.fit(X_train,
                            y_train,
                            batch_size = 256,
                            epochs = 200,
                            validation_data = (X_valid, y_valid),
                            callbacks = [es])
        train_time = time.time()-start_train

        start_test = time.time()
        proba_test = model.predict(X_test)
        test_time = time.time() - start_test

        test_size = X_test.shape[0]
        inf_time[k] = test_time/test_size

        aucs[k] = roc_auc_score(y_test, proba_test)
        print('S{0}, P{1} -- AUC: {2}'.format(subject, k, aucs[k]))
        K.clear_session()

    np.savetxt(modelpath + '/s' + str(subject) + '_auc.npy', aucs)
    np.savetxt(modelpath + '/inf_time.npy', inf_time)

    np.save(modelpath + '/s' + str(subject) + '_data.npy', X_test)
    np.save(modelpath + '/s' + str(subject) + '_labels.npy', y_test)
    model.save_weights(modelpath + '/s' + str(subject) + '_model.h5')



within_modelpath = "/workspace/data/EEG/models/sepconv1d/within/"

for i in range(8):
    evaluate_subject_models(tCE, lab, within_modelpath, i)

plot_within_aucs(within_modelpath, within_modelpath+"sepconv1d_within_auc.png", "SepConv1d - Within")