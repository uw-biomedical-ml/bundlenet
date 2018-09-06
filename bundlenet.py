"""

Functions for bundlenet: a convolutional neural network
for segmentation of human brain connectomes


"""

from glob import glob
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nibabel.streamlines import load as load_trk
import nibabel as nib

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from skimage.transform import resize
import scipy.ndimage.morphology as morph
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from numba import jit
import dask.array as da
from dask import delayed
import time


def read_sl(fname):
    """
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines,
                                   np.eye(4), tgram.affine))
    return sl

def binary_dilation(input, structure, iterations, origin):
    invert =1
    mask = None
    output = np.zeros(input.shape,bool)
    border_value = 0
    
    if iterations == 1:
        _nd_image.binary_erosion(input, structure, mask, output,
                                 border_value, origin, invert, True, 0)
        return output
    else:
        changed, coordinate_list = morph._nd_image.binary_erosion(
            input, structure, mask, output,
            border_value, origin, invert, True, 1)
        structure = structure[tuple([slice(None, None, -1)] *
                                    structure.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if not structure.shape[ii] & 1:
                origin[ii] -= 1
        morph._nd_image.binary_erosion2(output, structure, mask, iterations - 1,
                                  origin, invert, coordinate_list)
        return output
    
#a decorator    
@jit
def sparse_dilation(a):
    pos = np.where(a)

    for p in zip(pos[0], pos[1]):
        a[p[0] - 1 : p[0] + 2, p[1] - 1:p[1] + 2] = 1

    return a 
    
def reduce_sl(sl, vol_shape, structure, origin, dilation_iter=5, size=100):
    """
    Reduces a 3D streamline to a binarized 100 x 100 image.
    """ 
    #vol_shape = (vol_shape[0], vol_shape[1], vol_shape[1])
    vol = np.zeros(vol_shape, dtype=bool)
    sl = np.round(sl).astype(int).T
    vol[sl[0], sl[1], sl[2]] = 1
    # emphasize it a bit:
    tic = time.clock()
    #vol = binary_dilation(vol, structure, dilation_iter, origin)
    #vol = morph.binary_dilation(vol, iterations=dilation_iter)
    #vol = gaussian_filter(vol, sigma=(5, 5, 5), order=0)
    toc = time.clock()
    #print("time dilation" + str(toc - tic))
    #tic = time.clock()
    #vol = resize(vol, (size, size, size))
    #toc = time.clock()
    #print("time 1st resize" + str(toc - tic))
    p0 = resize(np.max(vol, 0),(vol_shape[0], vol_shape[0]))
    p1 = resize(np.max(vol, 1),(vol_shape[0], vol_shape[0]))
    p2 = np.max(vol, 2) 
    projected = np.concatenate((p0,p1,p2))
    #projected = morph.binary_dilation(projected, iterations=dilation_iter)
    #kernel = np.ones((3,3))
    #projected = fftconvolve(projected,kernel)
    #projected = np.concatenate([np.max(vol, dim) for dim in range(len(vol.shape))])
    tic = time.clock()
    projected = resize(projected, (size, size, 1)) #expects 3-d, like rgb
    #projected = sparse_dilation(projected)
    projected = morph.binary_dilation(projected, iterations=dilation_iter)
    toc = time.clock()
    #print("time 2nd resize" + str(toc - tic))
    return projected


def partition_data(bundle_files, vol_shape, take_n_bundles,
                   take_n_sl, dilation_iter=5, size=100):
    """
    Creates the data-sets for training, validation, testing.


    """
#     data_train = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.6)),
#                            size, size, 1), dtype='float32')
#     data_valid = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),
#                            size, size, 1), dtype='float32')
#     data_test = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),
#                           size, size, 1), dtype='float32')
#     data_train = da.from_array(data_train,chunks=data_train.shape)
#     data_valid = da.from_array(data_valid,chunks=data_valid.shape)
#     data_test = da.from_array(data_test,chunks=data_test.shape)

    data_train = []
    data_valid = []
    data_test = []
    
    labels_train = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.6)))
    labels_valid = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))
    labels_test = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))

    ii_train = 0
    ii_valid = 0
    ii_test = 0

    tract_id = 0
    
    #removed from binary_dilation
    structure = morph.generate_binary_structure(3, 1)
    origin = morph._ni_support._normalize_sequence(0, 3)
    structure = structure[tuple([slice(None, None, -1)] *
                                structure.ndim)]
    
    print("number of bundles = %s" % (len(bundle_files)))
    tic_t = time.clock()      
    for fname in bundle_files:
        print(fname)
        tic = time.clock()
        tract_id += 1
        if tract_id > take_n_bundles:
            break
        # Shuffle them in case they are ordered somehow:
        streamlines = read_sl(fname)
        np.random.shuffle(streamlines)
        choose_sl = streamlines[:take_n_sl]
        for sl_idx, sl in enumerate(choose_sl):
            tic = time.clock()
            if np.any(sl < 0):
                non_z = np.where(sl<0)
                num_neg = len(non_z[0])
                tmp = np.delete(sl,non_z[0],axis=0)
                sl = tmp
                if np.any(sl < 0):
                    print("There are some negative coordinates",
                      "in %s, track number: %s" %
                      (fname, sl_idx))
                    print("####################")
            projected = da.from_delayed(delayed(reduce_sl)(sl, vol_shape, structure, origin, dilation_iter,
                                  size=size),(100,100,1),np.float32)
            if sl_idx < (np.round(take_n_sl * 0.2)):
                #data_test[ii_test] = projected
                data_test.append(projected)
                labels_test[ii_test] = tract_id
                ii_test += 1
            elif sl_idx < (np.round(take_n_sl * 0.4)):
                #data_valid[ii_valid] = projected
                data_valid.append(projected)
                labels_valid[ii_valid] = tract_id
                ii_valid += 1
            else:
                #data_train[ii_train] = projected
                data_train.append(projected)
                labels_train[ii_train] = tract_id
                ii_train += 1
            toc = time.clock()
            #print("time to process 1 streamline " + str(toc - tic))
        toc = time.clock()
        #print(toc - tic)
    toc_t = time.clock()
    print("total time to process streamlines " + str(toc_t - tic_t))
    return (data_train, data_valid, data_test, labels_train, labels_valid,
            labels_test)


def plot_accuracy(training):
    accuracy = training.history['acc']
    val_accuracy = training.history['val_acc']
    epochs = range(len(accuracy))
    fig, ax = plt.subplots(1)
    ax.plot(epochs, accuracy, 'bo--', label='Training accuracy')
    ax.plot(epochs, val_accuracy, 'ro--', label='Validation accuracy')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (% correct)")
    fig.suptitle('Training and validation accuracy')
    plt.legend()
    return fig


def plot_streamlines(streamlines, color=None):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if color is not None:
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], c=color)
    else:
        ax.plot(sl[:, 0], sl[:, 1], sl[:, 2], c=color)

    ax.view_init(elev=0, azim=0)
    plt.axis("off")

    return fig

