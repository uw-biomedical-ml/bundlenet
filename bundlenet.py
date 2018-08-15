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
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import gaussian_filter
import time


def read_sl(fname):
    """
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines,
                                   np.eye(4), tgram.affine))
    return sl

def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0, origin=0,
                    brute_force=False):
    """
    Multi-dimensional binary dilation with the given structuring element.
    Parameters
    ----------
    input : array_like
        Binary array_like to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure : array_like, optional
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one.
    iterations : {int, float}, optional
        The dilation is repeated `iterations` times (one, by default).
        If iterations is less than 1, the dilation is repeated until the
        result does not change anymore.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (dilated)
        in the current iteration; if True all pixels are considered as
        candidates for dilation, regardless of what happened in the previous
        iteration. False by default.
    Returns
    -------
    binary_dilation : ndarray of bools
        Dilation of the input by the structuring element.
    See also
    --------
    grey_dilation, binary_erosion, binary_closing, binary_opening,
    generate_binary_structure
    Notes
    -----
    Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for expanding the shapes in an image. The binary
    dilation of an image by a structuring element is the locus of the points
    covered by the structuring element, when its center lies within the
    non-zero points of the image.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology
    Examples
    --------
    >>> from scipy import ndimage
    >>> a = np.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a)
    array([[False, False, False, False, False],
           [False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False],
           [False, False, False, False, False]], dtype=bool)
    >>> ndimage.binary_dilation(a).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # 3x3 structuring element with connectivity 1, used by default
    >>> struct1 = ndimage.generate_binary_structure(2, 1)
    >>> struct1
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # 3x3 structuring element with connectivity 2
    >>> struct2 = ndimage.generate_binary_structure(2, 2)
    >>> struct2
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct1,\\
    ... iterations=2).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    """
    
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(input, structure, iterations, mask,
                           output, border_value, origin, 1, brute_force)
    
    
def reduce_sl(sl, vol_shape, dilation_iter=5, size=100):
    """
    Reduces a 3D streamline to a binarized 100 x 100 image.
    """ 
    #vol_shape = (vol_shape[0], vol_shape[1], vol_shape[1])
    vol = np.zeros(vol_shape, dtype=bool)
    sl = np.round(sl).astype(int).T
    vol[sl[0], sl[1], sl[2]] = 1
    # emphasize it a bit:
    tic = time.clock()
    vol = binary_dilation(vol, iterations=dilation_iter)
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
    #projected = np.concatenate([np.max(vol, dim) for dim in range(len(vol.shape))])
    tic = time.clock()
    projected = resize(projected, (size, size, 1)) #expects 3-d, like rgb
    #projected = binary_dilation(projected, iterations=dilation_iter)
    toc = time.clock()
    #print("time 2nd resize" + str(toc - tic))
    return projected


def partition_data(bundle_files, vol_shape, take_n_bundles,
                   take_n_sl, dilation_iter=5, size=100):
    """
    Creates the data-sets for training, validation, testing.


    """
    data_train = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.6)),
                           size, size, 1), dtype='float32')
    data_valid = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),
                           size, size, 1), dtype='float32')
    data_test = np.zeros((np.int(np.round(take_n_bundles * take_n_sl * 0.2)),
                          size, size, 1), dtype='float32')

    labels_train = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.6)))
    labels_valid = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))
    labels_test = np.zeros(np.int(np.round(take_n_bundles * take_n_sl * 0.2)))

    ii_train = 0
    ii_valid = 0
    ii_test = 0

    tract_id = 0
    
    #removed from binary_dilation
    structure = generate_binary_structure(input.ndim, 1)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    structure = numpy.asarray(structure)
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
            projected = reduce_sl(sl, vol_shape, dilation_iter=dilation_iter,
                                  size=size)
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
            toc = time.clock()
            #print(toc - tic)
        toc = time.clock()
        #print(toc - tic)
    toc_t = time.clock()
    print("total time to process sreamlines " + str(toc_t - tic_t))
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
