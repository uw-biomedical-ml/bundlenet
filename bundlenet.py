"""

Functions for bundlenet: a convolutional neural network 
for segmentation of human brain connectomes


"""

from glob import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from nibabel.streamlines import load as load_trk
import nibabel as nib

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation


def read_sl(fname):
    """ 
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines, 
                                   np.eye(4), tgram.affine))
    return sl


def reduce_sl(sl, vol_shape, dilation_iter=5, size=100):
    """ 
    Reduces a 3D streamline to a binarized 100 x 100 image.
    """
    vol = np.zeros(vol_shape, dtype=bool)
    sl = np.round(sl).astype(int).T
    vol[sl[0], sl[1], sl[2]] = 1
    # emphasize it a bit:
    vol = binary_dilation(vol, iterations=dilation_iter)
    vol = resize(vol, (size, size, size))
    projected = np.concatenate([np.max(vol, dim) for dim in range(len(vol.shape))])
    projected = resize(projected, (size, size, 1))
    return projected


def partition_data(bundle_files, vol_shape, take_n_bundles, take_n_sl, dilation_iter=5, size=100):
    """ 
    Creates the data-sets for training, validation, testing.

    
    """
    data_train = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.6)),  size, size, 1), dtype='float32')
    data_valid = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),  size, size, 1), dtype='float32')
    data_test = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),  size, size, 1), dtype='float32')

    labels_train = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.6)))
    labels_valid = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))
    labels_test = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))

    ii_train = 0
    ii_valid = 0
    ii_test = 0

    tract_id = 0
    for fname in bundle_files:
        tract_id += 1
        if tract_id > take_n_bundles: 
            break
        # Shuffle them in case they are ordered somehow:
        streamlines = read_sl(fname) 
        np.random.shuffle(streamlines)
        choose_sl = streamlines[:take_n_sl]
        for sl_idx, sl in enumerate(choose_sl):
            if np.any(sl < 0):
                print("There are some negative coordinates in %s, track number: %s"%(fname, sl_idx))
            projected = reduce_sl(sl, vol_shape, dilation_iter=dilation_iter, size=size)
            if sl_idx < (np.round(take_n_sl * 0.2)):
                data_test[ii_test] = projected
                labels_test[ii_test] = tract_id
                ii_test += 1
            elif sl_idx < (np.round(take_n_sl * 0.4)):
                data_valid[ii_valid] = projected
                labels_valid[ii_valid] = tract_id
                ii_valid += 1
            else:
                data_train[ii_train] = projected
                labels_train[ii_train] = tract_id
                ii_train += 1
    return data_train, data_valid, data_test, labels_train, labels_valid, labels_test


def plot_accuracy(training):
    accuracy = training.history['acc']
    val_accuracy = training.history['val_acc']
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    epochs = range(len(accuracy))
    fig, ax = plt.subplots(1)
    ax.plot(epochs, accuracy, 'bo--', label='Training accuracy')
    ax.plot(epochs, val_accuracy, 'ro--', label='Validation accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (% correct)")
    fig.suptitle('Training and validation accuracy')
    plt.legend()
    return fig