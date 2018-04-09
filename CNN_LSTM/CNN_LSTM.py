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
n_train = n_example * 0.5
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

layer1 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv1", init='glorot_uniform', data_format="channels_first")(input_x_padding)
layer1 = Dropout(dr)(layer1)
layer1_padding = ZeroPadding2D((0, 2), data_format="channels_first")(layer1)

layer2 = Conv2D(50, (1, 8), padding="valid", activation="relu", name="conv2", init='glorot_uniform', data_format="channels_first")(layer1_padding)
layer2 = Dropout(dr)(layer2)
layer2 = ZeroPadding2D((0, 2), data_format="channels_first")(layer2)

layer3 = Conv2D(50, (1, 8), padding='valid', activation="relu", name="conv3", init='glorot_uniform', data_format="channels_first")(layer2)
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
epochs = 100  # number of epochs to train on
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


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)
    plt.show()


# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)

pre_labels = []
for x in test_Y_hat:
    tmp = np.argmax(x, 0)
    pre_labels.append(tmp)
true_labels = []
for x in Y_test:
    tmp = np.argmax(x, 0)
    true_labels.append(tmp)

kappa = cohen_kappa_score(pre_labels, true_labels)
oa = accuracy_score(true_labels, pre_labels)
kappa_oa = {}
print('oa_all:', oa)
print('kappa_all:', kappa)
kappa_oa['oa_all'] = oa
kappa_oa['kappa_all'] = kappa
fd = open('results_all_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, kappa_oa), fd)
fd.close()
cnf_matrix = confusion_matrix(true_labels, pre_labels)

conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] += 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, labels=classes, title='%s Confusion matrix' % (name))

acc = {}
kappa_dict = {}
oa_dict = {}
for snr in snrs:

    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    test_Y_i_hat = model.predict(test_X_i)

    pre_labels_i = []
    for x in test_Y_i_hat:
        tmp = np.argmax(x, 0)
        pre_labels_i.append(tmp)
    true_labels_i = []
    for x in test_Y_i:
        tmp = np.argmax(x, 0)
        true_labels_i.append(tmp)
    kappa = cohen_kappa_score(pre_labels_i, true_labels_i)
    oa = accuracy_score(true_labels_i, pre_labels_i)
    oa_dict[snr] = oa
    kappa_dict[snr] = kappa
    cnf_matrix = confusion_matrix(true_labels_i, pre_labels_i)
    np.set_printoptions(precision=2)
    plt.figure()
    plt.show()

    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] += 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="%s Confusion Matrix (SNR=%d)" % (name, snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

print 'acc:', acc
fd = open('results_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, acc), fd)
fd.close()
print('oa:', oa_dict)
fd = open('results_oa_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, oa_dict), fd)
fd.close()
print('kappa:', kappa_dict)
fd = open('results_kappa_%s_d0.5.dat' % (name), 'wb')
cPickle.dump(("%s" % (name), 0.5, kappa_dict), fd)
fd.close()

plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("%s Classification Accuracy on RadioML 2016.10 Alpha" % (name))
plt.savefig("%s Classification Accuracy" % (name))
plt.show()