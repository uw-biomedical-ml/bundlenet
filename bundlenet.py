"""

Functions for bundlenet: a convolutional neural network
for segmentation of human brain connectomes


"""

import numpy as np
from skimage.transform import resize
import scipy.ndimage.morphology as morph

from nibabel.streamlines import load as load_trk
import dipy.tracking.utils as dtu

from sklearn.metrics import cohen_kappa_score, jaccard_similarity_score,confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns


def read_sl(fname):
    
    """
    Reads streamlins from file.
    """
    tgram = load_trk(fname)
    sl = list(dtu.move_streamlines(tgram.streamlines,
                                   np.eye(4), tgram.affine))
    return sl

def process_sl(streamlines_tract,take_n_sl,vol_shape,size):
    if take_n_sl == -1 :
        take_n_sl = len(streamlines_tract)
    projected_all = np.zeros([take_n_sl,size,size,1])
    np.random.shuffle(streamlines_tract)
    resize_dim = max(vol_shape)
    s1_selected = streamlines_tract[:take_n_sl]
    for sl_idx, sl in enumerate(s1_selected):
        vol = np.zeros(vol_shape, dtype=bool)
        sl = np.round(sl).astype(int).T
        vol[sl[0], sl[1], sl[2]] = 1
        p0 = resize(np.max(vol, 0),(resize_dim,resize_dim))
        p1 = resize(np.max(vol, 1),(resize_dim,resize_dim))
        p2 = resize(np.max(vol, 2),(resize_dim,resize_dim)) 
        projected = np.concatenate((p0,p1,p2))
        projected = morph.binary_dilation(projected, iterations=5)
        projected = resize(projected, (size, size,1)) #expects 3-d, like rgb
        projected_all[sl_idx,:,:,:]=projected
    return projected_all

def partition_testtrain(test_perc, val_perc, streamlines_processed):
    all_streamlines = streamlines_processed[0]
    all_labels = np.zeros((streamlines_processed[0].shape[0]))
    for i in range(1,len(streamlines_processed)):
        all_streamlines = np.concatenate((all_streamlines,streamlines_processed[i]),axis=0)
        all_labels = np.concatenate((all_labels,i*np.ones((streamlines_processed[i].shape[0]))))
    data_trainval, data_test, labels_trainval, labels_test = train_test_split(all_streamlines, all_labels, test_size=test_perc, stratify=all_labels)
    data_train, data_val, labels_train, labels_val = train_test_split(data_trainval, labels_trainval, test_size=val_perc/(1-test_perc), stratify=labels_trainval)
    return (data_train, data_test, data_val, labels_train, labels_test, labels_val)

def getdata(streamlines_processed):
    all_streamlines = streamlines_processed[0]
    all_labels = np.zeros((streamlines_processed[0].shape[0]))
    for i in range(1,len(streamlines_processed)):
        all_streamlines = np.concatenate((all_streamlines,streamlines_processed[i]),axis=0)
        all_labels = np.concatenate((all_labels,i*np.ones((streamlines_processed[i].shape[0]))))
    return (all_streamlines,all_labels)

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

def print_accuarcystats(p_idx,labels_actual_idx):
    print("Percent correct is %s " % np.mean(p_idx == labels_actual_idx))
    kappa = cohen_kappa_score(p_idx, labels_actual_idx)
    print("Kappa is: %s" % kappa)
    jaccard = jaccard_similarity_score(p_idx, labels_actual_idx)
    print("Jaccard is: %s" % jaccard)

def plotconfusionmat(bundle_names,p_idx,labels_actual_idx):
    labels = np.array(range(min(p_idx),max(p_idx)+1))
    confusion_mat = confusion_matrix(labels_actual_idx,p_idx, labels)
    arr_bundle_names = np.array(bundle_names)
    sort_idx = np.argsort(arr_bundle_names)
    fig, ax = plt.subplots(1)
    sns.heatmap(confusion_mat[sort_idx][:, sort_idx],
            xticklabels=arr_bundle_names[sort_idx], 
            yticklabels=arr_bundle_names[sort_idx], ax=ax)
    fig.set_size_inches([10, 8])