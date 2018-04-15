import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import sys
sys.path.append('../modulation')
import numpy as np
import matplotlib.pyplot as plt
import keras, cPickle
from keras.layers import LSTM, Input
from keras.models import Model
from keras.layers.core import Reshape, Dropout, Dense, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score

name = 'CNN_LSTM'

Xd = cPickle.load(open("RML2016.10a_dict.dat", 'rb'))
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))

X = np.vstack(X)
np.random.seed(1567)
n_example = X.shape[0]
n_train = n_example * 0.7
train_idx = np.random.choice(range(0,n_example), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_example)) - set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]

def to_onehot(vec):
    vec_hot = np.zeros([len(vec), max(vec) + 1])
    vec_hot[np.arange(len(vec)), vec] = 1
    return vec_hot


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods

dr = 0.5  # dropout rate l

# Reshape [N,2,128] to [N,1,2,128] on input
input_x = Input(shape=(1, 2, 128))

# channels_first corresponds to inputs with shape (batch, channels, height, width).
# Build our MOdel
input_x_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_x)

layer1 = Conv2D(50, (1, 7), padding='valid', activation="relu", name="conv1", init='glorot_uniform', data_format="channels_first")(input_x_padding)
layer1 = Dropout(dr)(layer1)
layer1_padding = ZeroPadding2D((0, 2), data_format="channels_first")(layer1)

layer2 = Conv2D(50, (1, 7), padding="valid", activation="relu", name="conv2", init='glorot_uniform', data_format="channels_first")(layer1_padding)
layer2 = Dropout(dr)(layer2)
layer2 = ZeroPadding2D((0, 2), data_format="channels_first")(layer2)

layer3 = Conv2D(50, (1, 7), padding='valid', activation="relu", name="conv3", init='glorot_uniform', data_format="channels_first")(layer2)
layer3 = Dropout(dr)(layer3)

concat = keras.layers.concatenate([layer1, layer3])
concat_size = list(np.shape(concat))
input_dim = int(concat_size[-1] * concat_size[-2])
timesteps = int(concat_size[-3])
concat = Reshape((timesteps, input_dim))(concat)
lstm_out = LSTM(50, input_dim=input_dim, input_length=timesteps)(concat)
layer_dense1 = Dense(256, activation='relu', init='he_normal', name="dense1")(lstm_out)
layer_dropout = Dropout(dr)(layer_dense1)
layer_dense2 = Dense(len(classes), init='he_normal', name="dense2")(layer_dropout)
layer_softmax = Activation('softmax')(layer_dense2)

output = Reshape([len(classes)])(layer_softmax)

model = Model(inputs=input_x, outputs=output)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# End of building, we will start fitting the neural network
# Set up some params
epochs = 150  # number of epochs to train on
batch_size = 1024  # training batch size default1024
filepath = "convmodrecnets_%s_0.5.wts.h5" % ('CNN_LSTM')

X_train = np.reshape(X_train, (-1, 1, 2, 128))
X_test = np.reshape(X_test, (-1, 1, 2, 128))

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, Y_test))
# Show loss curves
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('%s Training performance' %(name))
# plt.show()

model.load_weights(filepath)
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('evaluate_score:', score)
