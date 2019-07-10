"""

Functions for bundlenet: a convolutional neural network
for segmentation of tractography streamlines


"""

import numpy as np
from skimage.transform import resize
import scipy.ndimage.morphology as morph

import dipy.tracking.utils as dtu

from sklearn.metrics import cohen_kappa_score, jaccard_similarity_score,confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D

from sklearn.utils import class_weight

from keras.preprocessing.image import ImageDataGenerator

from dipy.io.streamline import load_trk, save_trk
from dipy.tracking.streamline import Streamlines

import xgboost as xgb

def read_sl(fname):    
    """
    Reads streamlines from file.
    """
    streams, hdr = load_trk(fname)
    sl = Streamlines(streams)
    
    return sl

def read_sl_mni(fname):    
    """
    Reads streamlines from file in MNI space.
    """
    streams, hdr = load_trk(fname)
    sl = Streamlines(streams)
    sl_mni=[]
    for i in range(len(sl)):
        tmp = sl[i]
        tmp2=np.zeros([len(tmp),3])
        tmp2[:,0] = tmp[:,0] * -1 + 90
        tmp2[:,1] = tmp[:,1] + 126
        tmp2[:,2] = tmp[:,2] + 72
        sl_mni.append(np.round(tmp2))
    return sl_mni


def process_sl(streamlines_tract,take_n_sl,vol_shape,size,dil_iters): 
    """
    Takes dask bag of loaded bundles and returns sizexsize MIP image
    """  
    if take_n_sl == -1 or take_n_sl > len(streamlines_tract):
        take_n_sl = len(streamlines_tract)
    else:
        np.random.shuffle(streamlines_tract)
    
    projected_all = np.zeros([take_n_sl,size,size,1])
    
    resize_dim = size #min(vol_shape)
    s1_selected = streamlines_tract[:take_n_sl]
    for sl_idx, sl in enumerate(s1_selected):
        if sl_idx % 1000 == 0:
            print(sl_idx)
        vol = np.zeros(vol_shape, dtype=bool)
        sl = np.round(sl).astype(int).T
        vol[sl[0], sl[1], sl[2]] = 1
        p0 = resize(np.max(vol, 0),(resize_dim,resize_dim))
        p1 = resize(np.max(vol, 1),(resize_dim,resize_dim))
        p2 = resize(np.max(vol, 2),(resize_dim,resize_dim)) 
        projected = np.concatenate((p0,p1,p2))
        if dil_iters != 0:
            projected = morph.binary_dilation(projected, iterations=dil_iters)
        projected = resize(projected, (size, size,1)) #expects 3-d, like rgb
        projected_all[sl_idx,:,:,:]=projected
    return projected_all

def process_sl_onedirection(streamlines_tract,take_n_sl,vol_shape,size,dil_iters): 
    """
    Takes dask bag of loaded bundles and returns sizexsize MIP image splitting by x,y,z
    """ 
    if take_n_sl == -1 or take_n_sl > len(streamlines_tract):
        take_n_sl = len(streamlines_tract)
    else:
        np.random.shuffle(streamlines_tract)
    
    projected_all_0 = np.zeros([take_n_sl,size,size,1])
    projected_all_1 = np.zeros([take_n_sl,size,size,1])
    projected_all_2 = np.zeros([take_n_sl,size,size,1])
    
    resize_dim = max(vol_shape)
    s1_selected = streamlines_tract[:take_n_sl]
    for sl_idx, sl in enumerate(s1_selected):
        if sl_idx % 1000 == 0:
            print(sl_idx)
        vol = np.zeros(vol_shape, dtype=bool)
        sl = np.round(sl).astype(int).T
        vol[sl[0], sl[1], sl[2]] = 1
        p0 = resize(np.max(vol, 0),(size,size,1))
        p1 = resize(np.max(vol, 1),(size,size,1))
        p2 = resize(np.max(vol, 2),(size,size,1)) 
        p0_dil = morph.binary_dilation(p0, iterations=dil_iters)
        p1_dil = morph.binary_dilation(p1, iterations=dil_iters)
        p2_dil = morph.binary_dilation(p2, iterations=dil_iters)
        projected_all_0[sl_idx,:,:,:]=p0_dil
        projected_all_1[sl_idx,:,:,:]=p1_dil
        projected_all_2[sl_idx,:,:,:]=p2_dil
    return(projected_all_0, projected_all_1, projected_all_2)

def partition_testtrain(test_perc, val_perc, streamlines_processed):
    
    """
    Partitions data into test, train, and validation
    """
    all_streamlines = streamlines_processed[0]
    all_labels = np.zeros((streamlines_processed[0].shape[0]))
    for i in range(1,len(streamlines_processed)):
        all_streamlines = np.concatenate((all_streamlines,streamlines_processed[i]),axis=0)
        all_labels = np.concatenate((all_labels,i*np.ones((streamlines_processed[i].shape[0]))))
    if test_perc > 0:
        data_trainval, data_test, labels_trainval, labels_test = train_test_split(all_streamlines, all_labels, test_size=test_perc, stratify=all_labels)
    else:
        data_trainval = all_streamlines
        data_test = []
        labels_trainval = all_labels
        labels_test = []
    if val_perc > 0:
        data_train, data_val, labels_train, labels_val = train_test_split(data_trainval, labels_trainval, test_size=val_perc/(1-test_perc), stratify=labels_trainval)
    else:
        data_train = data_trainval
        labels_train = labels_trainval
        data_val = []
        labels_val = []
    return (data_train, data_test, data_val, labels_train, labels_test, labels_val)

def partition_testtrain_onedirection(test_perc, val_perc, streamlines_processed, dim_proj):
    """
    Partitions data into test, train, and validation with 1 direction MIP images
    """
    all_streamlines = streamlines_processed[0][dim_proj]
    all_labels = np.zeros((streamlines_processed[0][dim_proj].shape[0]))
    for i in range(1,len(streamlines_processed)):
        all_streamlines = np.concatenate((all_streamlines,streamlines_processed[i][dim_proj]),axis=0)
        all_labels = np.concatenate((all_labels,i*np.ones((streamlines_processed[i][dim_proj].shape[0]))))
    if test_perc > 0:
        data_trainval, data_test, labels_trainval, labels_test = train_test_split(all_streamlines, all_labels, test_size=test_perc, stratify=all_labels)
    else:
        data_trainval = all_streamlines
        data_test = []
        labels_trainval = all_labels
        labels_test = []
    if val_perc > 0:
        data_train, data_val, labels_train, labels_val = train_test_split(data_trainval, labels_trainval, test_size=val_perc/(1-test_perc), stratify=labels_trainval)
    else:
        data_train = data_trainval
        labels_train = labels_trainval
        data_val = []
        labels_val = []
    return (data_train, data_test, data_val, labels_train, labels_test, labels_val)

def buildmodel_specify(input_shape, num_classes, dropout_factor, num_convlayers, num_fulllayers):
    """
    Constructs a CNN given specified parameters (num_classes,dropout_factor, num_convlayers, num_fulllayers)
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape,padding='same'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(dropout_factor))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(dropout_factor))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(dropout_factor))
    if num_convlayers > 3:
        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(dropout_factor))
    if num_convlayers > 4:
        model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
        model.add(Dropout(dropout_factor))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_factor))
    if num_fulllayers > 1:
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_factor))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    return model

def getlabeledstreamlines(map_files):
    """
    Returns the streamlines with a label from a list of trk files
    """
    labeled_index = []
    labels = []
    for m_idx, m in enumerate(map_files):
        tmp = np.load(m)
        labeled_index = np.append(labeled_index,tmp)
        labels = np.append(labels,m_idx*np.ones([len(tmp),1]))
    return(labeled_index, labels) 
    
def getunlabeledstreamlines(n_sl, labeled_index, n_unlabeled, randomize_sl):
    """
    Returns the streamlines without a label given a list with labels
    """
    ind = range(n_sl)
    ind = np.delete(ind,labeled_index)
    if randomize_sl == 1:
        np.random.shuffle(ind)
    unlabeled_index = ind[0:n_unlabeled]
    return(unlabeled_index)

def combinestreamlines(labeled_index, unlabeled_index, labels, streamlines):
    """
    Combines lists of streamlines
    """
    labels_selected = np.append(labels,(np.max(labels)+1)*np.ones([len(unlabeled_index),1]))
    streamlines_selected = [streamlines[i] for i in np.int_(np.append(labeled_index, unlabeled_index))]
    return(labels_selected, streamlines_selected)

def getdatafrombag(streamlines_processed):   
    """
    Makes an nd.array of all streamlines from dask bag, used when not partitioning
    """
    all_streamlines = streamlines_processed[0]
    all_labels = np.zeros((streamlines_processed[0].shape[0]))
    for i in range(1,len(streamlines_processed)):
        all_streamlines = np.concatenate((all_streamlines,streamlines_processed[i]),axis=0)
        all_labels = np.concatenate((all_labels,i*np.ones((streamlines_processed[i].shape[0]))))
    return (all_streamlines,all_labels)

def plot_accuracy(training):  
    """
    Plots accuracy stats.
    """  
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

def plot_loss(training):   
    """
    Plots loss stats.
    """
    loss = training.history['loss']
    val_loss = training.history['val_loss']
    epochs = range(len(loss))
    fig, ax = plt.subplots(1)
    ax.plot(epochs, loss, 'bo--', label='Training loss')
    ax.plot(epochs, val_loss, 'ro--', label='Validation loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.suptitle('Training and validation loss')
    plt.legend()
    return fig

def print_accuarcystats(p_idx,labels_actual_idx):    
    """
    Prints accuracy stats.
    """ 
    print("Percent correct is %s " % np.mean(p_idx == labels_actual_idx))
    kappa = cohen_kappa_score(p_idx, labels_actual_idx)
    print("Kappa is: %s" % kappa)
    jaccard = jaccard_similarity_score(p_idx, labels_actual_idx)
    print("Jaccard is: %s" % jaccard)

def plotconfusionmat(bundle_names,p_idx,labels_actual_idx):    
    """
    Calculates and plots confusion matrix.
    """  
    labels = np.array(range(min(p_idx),max(p_idx)+1))
    confusion_mat = confusion_matrix(labels_actual_idx,p_idx, labels)
    arr_bundle_names = np.array(bundle_names)
    sort_idx = np.argsort(arr_bundle_names)
    fig, ax = plt.subplots(1)
    sns.heatmap(confusion_mat[sort_idx][:, sort_idx],
            xticklabels=arr_bundle_names[sort_idx], 
            yticklabels=arr_bundle_names[sort_idx], ax=ax)
    fig.set_size_inches([10, 8])
    
def run_xgboost(X_train,y_train,X_test,y_test,max_depth,num_class,thresh):
    """
    Runs XGBoost on test and train data
    """  
    if num_class < 3:
        obj = 'binary:logistic'
        param = {
        'max_depth': max_depth,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': obj,  # error evaluation for multiclass training
        }  # the number of classes that exist in this datset

    else:
        obj= 'multi:softprob'
        param = {
        'max_depth': max_depth,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': obj,  # error evaluation for multiclass training
        'num_class': num_class}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    bst = xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    if obj.startswith('multi'):
        p = np.argmax(preds,axis=1)
    else:
        p = (preds > thresh).astype(int)
    return p
    

def trainmodel(data_train, labels_train, labels_train_o, data_val, labels_val, data_test, labels_test, input_shape, num_classes, batch_size, epochs, data_aug, num_samples): 
    """
    Train CNN with the option of data augmentation (out of date)
    """ 
    model = buildmodel_specify(input_shape, num_classes, dropout_factor, num_convlayers, num_fulllayers)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
    
    #class_weights = class_weight.compute_class_weight('balanced',
                                                      #np.unique(labels_train_o),
                                                      #labels_train_o)
    if data_aug == 0: 
        training = model.fit(data_train, labels_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(data_val, labels_val))#,
                     #class_weight=class_weights) 
        fig = plot_accuracy(training)
    else:
        #need to shuffle streamlines and labels
        s = np.arange(len(labels_train))
        np.random.shuffle(s)
        data_train_shuff = data_train[s,:,:,:]
        labels_train_shuff = labels_train[s,:]
        
        steps_per_epoch = num_samples
        validation_steps = num_samples

        datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=5,
        height_shift_range=5,
        validation_split=0.2)
    
        training = model.fit_generator(datagen.flow(data_train_shuff, labels_train_shuff, batch_size=batch_size, subset='training'),
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=datagen.flow(data_train_shuff,labels_train_shuff, batch_size=batch_size, subset='validation'),
                    validation_steps=validation_steps)
        fig = plot_accuracy(training)
        fig = plot_loss(training)
    return model

def mniback(sl):
    """
    Transform from MNI space back 
    """ 
    sl_mni=[]
    for i in range(len(sl)):
        tmp = sl[i]
        tmp2=np.zeros([len(tmp),3])
        tmp2[:,0] = (tmp[:,0] -90 )* -1 
        tmp2[:,1] = tmp[:,1] - 126
        tmp2[:,2] = tmp[:,2] - 72
        sl_mni.append(np.round(tmp2))
    return sl_mni

def savesegtrk(sl, k, classassign, c, p, p_thresh,filename,mni):
    """
    Save .trk files given streamlines to keep(k) and most probable bundle(classassign) and associated probability(p)
    """ 
    tmp = np.where((classassign==c) & (p > p_thresh) & (k==0))
    t = tmp[0]
    sl_trk = [ sl[i] for i in t]
    if mni == 1:
        sl_trk_mni = mniback(sl_trk)
    else:
        sl_trk_mni = sl_trk
    save_trk(filename, streamlines=sl_trk_mni, affine=np.eye(4))
