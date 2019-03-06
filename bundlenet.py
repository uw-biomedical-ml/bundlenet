"""

Functions for bundlenet: a convolutional neural network
for segmentation of human brain connectomes


"""

import numpy as np
from skimage.transform import resize
import scipy.ndimage.morphology as morph

#from nibabel.streamlines import load as load_trk
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

def read_sl(fname):
    
    """
    Reads streamlines from file.
    """
    streams, hdr = load_trk(fname)
    sl = Streamlines(streams)
    #tgram = load_trk(fname)
    #sl = list(dtu.move_streamlines(tgram.streamlines,
                                   #np.eye(4), tgram.affine))
    return sl

def read_sl_mni(fname):
    
    """
    Reads streamlines from file.
    """
    #tgram = load_trk(fname)
    #sl = list(dtu.move_streamlines(tgram.streamlines,
                                   #np.eye(4), tgram.affine))
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
    
    resize_dim = max(vol_shape)
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

def process_slandpredict(streamlines_tract,take_n_sl,vol_shape,size,dil_iters,model):
    
    """
    Takes dask bag of loaded bundles and returns sizexsize MIP image
    """
    projected_all = process_slandpredict(streamlines_tract,take_n_sl,vol_shape,size,dil_iters)
    p_sl = model.predict(projected_all, batch_size=5)
    return p_sl
    

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

def partition_testtrain_onedirection(test_perc, val_perc, streamlines_processed,dim_proj):
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
    Plots accuracy stats.
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
    Calculated and plots confusion matrix.
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
    

def trainmodel(data_train, labels_train, labels_train_o, data_val, labels_val, data_test, labels_test, input_shape, num_classes, batch_size, epochs, data_aug, num_samples): 
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=input_shape,padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(Dropout(0.25))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    
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
    
    sl_mni=[]
    for i in range(len(sl)):
        tmp = sl[i]
        tmp2=np.zeros([len(tmp),3])
        tmp2[:,0] = (tmp[:,0] -90 )* -1 
        tmp2[:,1] = tmp[:,1] - 126
        tmp2[:,2] = tmp[:,2] - 72
        sl_mni.append(np.round(tmp2))
    return sl_mni

def savesegtrk(sl, classassign, c, p, p_thresh,filename,mni):
    tmp = np.where((classassign==c) & (p > p_thresh))
    t = tmp[0]
    sl_trk = [ sl[i] for i in t]
    if mni == 1:
        sl_trk_mni = mniback(sl_trk)
    else:
        sl_trk_mni = sl_trk
    save_trk(filename, streamlines=sl_trk_mni, affine=np.eye(4))
