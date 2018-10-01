"""

Functions for bundlenet: a convolutional neural network
for segmentation of human brain connectomes


"""

import numpy as np
from skimage.transform import resize
import scipy.ndimage.morphology as morph

from nibabel.streamlines import load as load_trk
import dipy.tracking.utils as dtu

from sklearn.metrics import cohen_kappa_score, jaccard_similarity_score

import matplotlib.pyplot as plt


def read_sl(fname):
    
    """
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines,
                                   np.eye(4), tgram.affine))
    return sl

def process_sl(streamlines_tract,take_n_sl,vol_shape,size):
    projected_all = np.zeros([take_n_sl,size,size,1])
    np.random.shuffle(streamlines_tract)
    s1_selected = streamlines_tract[:take_n_sl]
    for sl_idx, sl in enumerate(s1_selected):
        vol = np.zeros(vol_shape, dtype=bool)
        sl = np.round(sl).astype(int).T
        vol[sl[0], sl[1], sl[2]] = 1
        p0 = resize(np.max(vol, 0),(vol_shape[0], vol_shape[0]))
        p1 = resize(np.max(vol, 1),(vol_shape[0], vol_shape[0]))
        p2 = np.max(vol, 2) 
        projected = np.concatenate((p0,p1,p2))
        projected = morph.binary_dilation(projected, iterations=5)
        projected = resize(projected, (size, size,1)) #expects 3-d, like rgb
        projected_all[sl_idx,:,:,:]=projected
    return projected_all

def partition_testtrain(test_perc, val_perc, streamlines_processed):
    take_n_sl = streamlines_processed[0].shape[0]
    take_n_bundles = len(streamlines_processed)
    size_slimage = streamlines_processed[0].shape[1]
    sl_fortrain = int(round(take_n_sl*(1-test_perc-val_perc)))
    sl_fortest = int(round(take_n_sl*(test_perc)))
    sl_forval = int(round(take_n_sl*(val_perc)))

    
    data_train = np.zeros((np.int(sl_fortrain*take_n_bundles),
                           size_slimage,size_slimage,1), dtype='float32')
    data_test = np.zeros((np.int(sl_fortest*take_n_bundles),
                           size_slimage,size_slimage,1), dtype='float32')
    data_val = np.zeros((np.int(sl_forval*take_n_bundles),
                           size_slimage,size_slimage,1), dtype='float32')
    labels_train= np.zeros(np.int(sl_fortrain*take_n_bundles), dtype='float32')
    labels_test= np.zeros(np.int(sl_fortest*take_n_bundles), dtype='float32')
    labels_val= np.zeros(np.int(sl_forval*take_n_bundles), dtype='float32')

                      
    for sidx, s in enumerate(streamlines_processed):
        data_train[sidx*sl_fortrain:(sidx+1)*sl_fortrain,:,:,:] = s[0:sl_fortrain]
        data_test[sidx*sl_fortest:(sidx+1)*sl_fortest,:,:,:] = s[sl_fortrain:sl_fortrain+sl_fortest]
        data_val[sidx*sl_forval:(sidx+1)*sl_forval,:,:,:] = s[sl_fortrain+sl_fortest:sl_fortrain+sl_fortest+sl_forval]

        labels_train[sidx*sl_fortrain:(sidx+1)*sl_fortrain]=sidx
        labels_test[sidx*sl_fortest:(sidx+1)*sl_fortest]=sidx
        labels_val[sidx*sl_forval:(sidx+1)*sl_forval]=sidx
    return (data_train, data_test, data_val labels_train, labels_test, labels_val)

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

def print_accuarcystats(p,labels_test,bundle_names):
    p_idx = np.argmax(p, axis=-1)
    p_bundles = np.array([bundle_names[ii] for ii in p_idx])
    loaded_from_file = np.load('./subject1_bundles.npz')
    labels_test = loaded_from_file['labels_test']
    actual_labels = np.array([bundle_names[ii] for ii in (labels_test - 1).astype(int)])
    print("Percent correct is %s " % np.mean(p_bundles == actual_labels))
    kappa = cohen_kappa_score(p_bundles, actual_labels)
    print("Kappa is: %s" % kappa)
    jaccard = jaccard_similarity_score(p_bundles, actual_labels)
    print("Jaccard is: %s" % jaccard)