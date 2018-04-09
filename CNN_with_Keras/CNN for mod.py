# importing the necessary packages
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.layers.core import Reshape, Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.noise import GaussianNoise
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import random, cPickle, keras
import sys
sys.path.append('../modulation')
import seaborn as sb

# load the RadioMl.10A data
Xd = cPickle.load(open("RML2016.10a_dict.dat", 'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)

# Split the data into train, Validation and test data
# initialize randomly the seed
np.random.seed(1567)
n_example = X.shape[0]
n_train = n_example * 0.5
train_idx = np.random.choice(range(0,n_example), size= int(n_train), replace=False)
test_idx = list(set(range(0, n_example)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]


#here we will code the vectors into one hot vectors
def to_onehot(vec):
    vec_hot = np.zeros([len(vec), max(vec) + 1])
    vec_hot[np.arange(len(vec)), vec] = 1
    return vec_hot


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods

# build the model
dr = 0.5   # dropout rate 50 %
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(256, (1, 3), border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(80, (1, 3), border_mode='valid', activation="relu", name="conv3", init='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Convolution2D(80, (1, 3), border_mode='valid', activation="relu", name="conv4", init='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(128, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), init='he_normal', name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Set up some params
nb_epoch = 100     # number of epochs to train on
batch_size = 1024  # training batch size
# perform training ...
#   - call the main training loop in keras for our network+dataset
filepath = 'CNN.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks =[
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only = True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)

# evaluate the performance of the neural network
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)

# Show loss curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('%s Training performance' %('CNN'))

