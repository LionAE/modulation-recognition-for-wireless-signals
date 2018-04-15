import os
import cPickle, keras
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
sys.path.append('../modulation')
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Reshape, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.regularizers import *
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, BatchNormalization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score

name = 'ResNet'

# the following function is learning an approximation of the identity function from the data
# By trying to figure out that the output will be equal to input similar to Autoencoders
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = ZeroPadding2D((0, 2), data_format="channels_first")(input_tensor)
    x = Conv2D(filters1, (1, 5), name=conv_name_base + '2a',data_format="channels_first", init='glorot_uniform')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters2, kernel_size,
               padding='valid', name=conv_name_base + '2b',data_format="channels_first", init='glorot_uniform')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c',data_format="channels_first", init='glorot_uniform')(x)  # (1, 1)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Dropout(dr)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a convolutionnal layer at shortcut"""
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    input_tensor_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_tensor)
    x = Conv2D(filters1, (1, 1),  # (1, 1)
               name=conv_name_base + '2a',data_format="channels_first",padding='valid', init='glorot_uniform')(input_tensor_padding)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters2, kernel_size, padding='valid',
               name=conv_name_base + '2b',data_format="channels_first", init='glorot_uniform')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c',data_format="channels_first", padding='valid', init='glorot_uniform')(x)  # (1, 1)
    x = Dropout(dr)(x)

    shortcut = Conv2D(filters3, (1, 1),   # (1, 1)
                      name=conv_name_base + '1',data_format="channels_first", init='glorot_uniform')(input_tensor_padding)
    shortcut = Dropout(dr)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


with open("RML2016.10a_dict.dat", 'rb') as xd1:
    Xd = cPickle.load(xd1)
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
X_train = X[train_idx]
X_test = X[test_idx]

#here we will code the vectors into one hot vectors
def to_onehot(vec):
    vec_hot = np.zeros([len(vec), max(vec) + 1])
    vec_hot[np.arange(len(vec)), vec] = 1
    return vec_hot


trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
dr = 0.5
# Reshape [N,2,128] to [N,1,2,128] on input
input_x = Input(shape=(1,2,128))

x = conv_block(input_x, (1, 7), [50, 50, 50], stage=1, block='a')
x = identity_block(x, (1, 7), [50, 50, 50], stage=1, block='b')

layer_Flatten = Flatten()(x)
layer_dense1 = Dense(256, activation='relu', init='he_normal', name="dense1")(layer_Flatten)
layer_dropout = Dropout(dr)(layer_dense1)
layer_dense2 = Dense(len(classes), init='he_normal', name="dense2")(layer_dropout)
layer_softmax = Activation('softmax')(layer_dense2)
output = Reshape([len(classes)])(layer_softmax)

model = Model(inputs=input_x, outputs=output)
myadam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=myadam)
model.summary()

# Set up  params
epochs = 100  # number of epochs to train on
batch_size = 1024  # training batch size default 1024
filepath = "convmodrecnets_%s_0.5.wts.h5" % (name)

X_train = np.reshape(X_train, (-1,1,2,128))
X_test = np.reshape(X_test, (-1,1,2,128))

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, Y_test))

plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('%s Training performance' %(name))
plt.show()

model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('evaluate_score:', score)