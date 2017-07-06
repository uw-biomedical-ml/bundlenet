import os
import sys
import os.path as op
from glob import glob


import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import (Convolution2D, MaxPooling2D, Convolution3D,
                          MaxPooling3D)

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K


import numpy as np

import dipy.data as dpd
import nibabel as nib


dpd.fetch_bundles_2_subjects()

DATA_SUB1 = '/home/arokem/.dipy/exp_bundles_and_maps/bundles_2_subjects/subj_1'

bundle_fnames = glob(op.join(DATA_SUB1, 'bundles/*.trk'))

t1_warped_img = nib.load(op.join(DATA_SUB1, 't1_warped.nii.gz'))

sl_sum = 0
for b_idx, bundle in enumerate(bundle_fnames):
    tgram = nib.streamlines.load(op.join(DATA_SUB1, 'bundles', bundle))
    print(bundle, len(tgram.streamlines))
    sl_sum += len(tgram.streamlines)

bundle_arr = np.memmap('/home/arokem/bundlenet/data/bundles/npy/bundles.npy', mode='r', 
                       shape=((sl_sum, ) + t1_warped_img.shape + (1,)), dtype=bool)
one_hot_arr = np.memmap('/home/arokem/bundlenet/data/bundles/npy/one_hot.npy', mode='r', 
                        shape=(sl_sum, len(bundle_fnames)), dtype=bool)



## For now, let's look at a tiny subset: 
bundle_arr = bundle_arr[::100]
one_hot_arr = one_hot_arr[::100]

batch_size = 1
num_classes = 27
epochs = 12

# input image dimensions
img_x, img_y, img_z = t1_warped_img.shape

import sklearn.model_selection as ms 

sp = ms.ShuffleSplit(n_splits=1, test_size=0.15)


for train_index, test_index in sp.split(bundle_arr):
    tr = train_index
    te = test_index



import tempfile 

tmpdir = tempfile.TemporaryDirectory()

bundle_train_file = op.join(tmpdir.name, 'bundles_train.npy')
bundle_test_file = op.join(tmpdir.name, 'bundles_test.npy')
one_hot_train_file = op.join(tmpdir.name, 'one_hot_train.npy')
one_hot_test_file = op.join(tmpdir.name, 'one_hot_test.npy')



bundle_train_arr = np.zeros((tr.shape + t1_warped_img.shape + (1,)), dtype=bool)
bundle_test_arr = np.zeros((te.shape + t1_warped_img.shape + (1,)), dtype=bool)
one_hot_train_arr = np.zeros((tr.shape[0], len(bundle_fnames)), dtype=bool)
one_hot_test_arr = np.zeros((te.shape[0], len(bundle_fnames)), dtype=bool)


for ii, idx in enumerate(tr):
    bundle_train_arr[ii] = bundle_arr[idx]
    one_hot_train_arr[ii] = one_hot_arr[idx]

for ii, idx in enumerate(te):
    bundle_test_arr[ii] = bundle_arr[idx]
    one_hot_test_arr[ii] = one_hot_arr[idx]


input_shape = (img_x, img_y, img_z, 1)


model = Sequential()
model.add(Convolution3D(32, kernel_size=(3, 3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Convolution3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


descript = "description-" 

datadir = "/home/arokem/bundlenet/data/"
rundir = "/home/arokem/bundlenet/run/"


class LossHistory(keras.callbacks.Callback):
    def __init__(self, directory, descript, augment):
        self.d = directory
        self.descript = descript
        self.augment = augment

    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("%s/history.txt" % self.d, "a") as fout:
                for metric in ["loss", "acc" ]:
                        fout.write("%s\t%d\ttrain\t%d\t%s\t%.6f\n" % (self.descript, self.augment, self.lastiter, metric, logs.get(metric)))
                for metric in ["val_loss", "val_acc"]:
                        fout.write("%s\t%d\tvalid\t%d\t%s\t%.6f\n" % (self.descript, self.augment, self.lastiter, metric, logs.get(metric)))

    def on_batch_end(self, batch, logs={}):
        self.lastiter += 1
        with open("%s/history.txt" % self.d, "a") as fout:
                for metric in ["loss", "acc"]:
                        fout.write("%s\t%d\ttrain\t%d\t%s\t%.6f\n" % (self.descript, self.augment, self.lastiter, metric, logs.get(metric)))


#rundir += descript
if not op.exists(rundir):
    os.mkdir("%s/"  % (rundir))
if not op.exists("%s/weights"  % (rundir)):
    os.mkdir("%s/weights"  % (rundir))


print(model.summary())
with open("%s/model.yaml" % (rundir), "w") as fout:
    fout.write(model.to_yaml())

plot_model(model, to_file='model.png')

checkpoint = ModelCheckpoint(rundir + 'check', save_best_only=True, mode='max', monitor='val_acc')

def generate_samples():
    while 1:
        for ii in range(bundle_train_arr.shape[0]):
            yield bundle_train_arr[ii:ii+1], one_hot_train_arr[ii:ii+1]


def generate_validation():
    while 1:
        for ii in range(bundle_test_arr.shape[0]):
            yield bundle_test_arr[ii:ii+1], one_hot_test_arr[ii:ii+1]


history = LossHistory(rundir, "model")
model.fit_generator(generate_samples(), 
                    nb_worker=1, 
                    samples_per_epoch=1,
                    validation_data=generate_validation(), 
                    nb_epoch=1, 
                    verbose=1, 
                    callbacks=[history, checkpoint])



