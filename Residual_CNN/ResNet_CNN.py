import os
import cPickle
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from keras.layers.core import Reshape, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.regularizers import *
import matplotlib.pyplot as plt
import pickle
import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense

import sys
sys.path.append('../confusion')
import plotcm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score

name = 'resnet'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = ZeroPadding2D((0, 2), data_format="channels_first")(input_tensor)
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',data_format="channels_first", init='glorot_uniform')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    # x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters2, kernel_size,
               padding='valid', name=conv_name_base + '2b',data_format="channels_first", init='glorot_uniform')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c',data_format="channels_first", init='glorot_uniform')(x)  # (1, 1)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Dropout(dr)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor_padding = ZeroPadding2D((0, 2), data_format="channels_first")(input_tensor)
    x = Conv2D(filters1, (1, 1),  # (1, 1)
               name=conv_name_base + '2a',data_format="channels_first",padding='valid', init='glorot_uniform')(input_tensor_padding)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    # 这里必须为same,要不然就会出现下面拼接的时候尺寸不对
    x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters2, kernel_size, padding='valid',
               name=conv_name_base + '2b',data_format="channels_first", init='glorot_uniform')(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Dropout(dr)(x)

    # x = ZeroPadding2D((0, 2), data_format="channels_first")(x)
    x = Conv2D(filters3, (1, 3), name=conv_name_base + '2c',data_format="channels_first", padding='valid', init='glorot_uniform')(x)  # (1, 1)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Dropout(dr)(x)

    shortcut = Conv2D(filters3, (1, 1),   # (1, 1)
                      name=conv_name_base + '1',data_format="channels_first", init='glorot_uniform')(input_tensor_padding)
    # shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Dropout(dr)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


# %%
with open("../RML2016.10a_dict.dat", 'rb') as xd1:  # 这段执行对原始数据进行切片的任务，可在spyder下运行，查看变量
    Xd = pickle.load(xd1) #, encoding='latin1'
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
# %%
np.random.seed(2016)  # 对预处理好的数据进行打包，制作成投入网络训练的格式，并进行one-hot编码
n_examples = X.shape[0]
n_train = n_examples * 0.5  # 对半
train_idx = np.random.choice(range(0, n_examples), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))  # label
X_train = X[train_idx]
X_test = X[test_idx]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


trainy = list(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_train = to_onehot(trainy)
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
# in_shp: <type 'list'>: [2, 128]
in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods
# %%
dr = 0.5  # dropout rate (%) 卷积层部分  https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/#conv2d

# 这里使用keras的函数式编程 http://keras-cn.readthedocs.io/en/latest/getting_started/functional_API/
# Reshape [N,2,128] to [N,1,2,128] on input
input_x = Input(shape=(1,2,128))

x = conv_block(input_x, (1,3), [50, 50, 50], stage=1, block='a')
x = identity_block(x, (1,7), [50, 50, 50], stage=1, block='b')
# x = identity_block(x, (1,9), [50, 50, 50], stage=1, block='c')

layer_Flatten = Flatten()(x)
layer_dense1 = Dense(256, activation='relu', init='he_normal', name="dense1")(layer_Flatten)
# 当使用原数据集的时候，输出为512的时候正确率更高
# layer_dense1 = Dense(512, activation='relu', init='he_normal', name="dense1")(layer_Flatten)
layer_dropout = Dropout(dr)(layer_dense1)
layer_dense2 = Dense(len(classes), init='he_normal', name="dense2")(layer_dropout)
layer_softmax = Activation('softmax')(layer_dense2)
output = Reshape([len(classes)])(layer_softmax)

model = Model(inputs=input_x, outputs=output)
myadam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=myadam)
model.summary()

# %%
# Set up some params
epochs = 100  # number of epochs to train on
batch_size = 1024  # training batch size default1024
# %%
filepath = "convmodrecnets_%s_0.5.wts.h5" % (name)   # 所要保存的文件名字，h5格式，不用写路径，默认在程序执行的文件夹内

X_train = np.reshape(X_train, (-1,1,2,128))
X_test = np.reshape(X_test, (-1,1,2,128))

history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    callbacks=[  # 回调函数，https://keras-cn.readthedocs.io/en/latest/other/callbacks/
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                                        mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])  # EarlyStopping 当监测值不再改善时，该回调函数将中止训练，如去除本行将执行所有epoch
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
    # plt.show()


# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)

# %%调用库产生混淆矩阵
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
# np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
# plt.figure()
plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                             normalize=False,
                             title='%s Confusion matrix, without normalization' % (name), showtext=True)
plt.savefig('%s Confusion matrix, without normalization' % (name))
# Plot normalized confusion matrix
# plt.figure()
plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                             normalize=True,
                             title='%s Normalized confusion matrix' % (name), showtext=True)
plt.savefig('%s Normalized confusion matrix' % (name))
# plt.show()

# %%自定义产生混淆矩阵
conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, X_test.shape[0]):
    j = list(Y_test[i, :]).index(1)
    k = int(np.argmax(test_Y_hat[i, :]))
    conf[j, k] += 1
for i in range(0, len(classes)):
    confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
plot_confusion_matrix(confnorm, labels=classes, title='%s Confusion matrix' % (name))

# %%Plot confusion matrix 画图
acc = {}
kappa_dict = {}
oa_dict = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)

    # %%调用库产生混淆矩阵
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
    # np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    # plt.figure()
    plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                                 normalize=False,
                                 title='%s Confusion matrix, without normalization (SNR=%d)' % (name, snr), showtext=True)
    plt.savefig('%s Confusion matrix, without normalization (SNR=%d)' % (name, snr))
    # Plot normalized confusion matrix
    # plt.figure()
    plotcm.plot_confusion_matrix(cnf_matrix, classes=classes,
                                 normalize=True,
                                 title='%s Normalized confusion matrix (SNR=%d)' % (name, snr), showtext=True)
    plt.savefig('%s Normalized confusion matrix (SNR=%d)' % (name, snr))
    # plt.show()

    # %%自定义产生混淆矩阵
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] += 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    # plt.figure()
    plot_confusion_matrix(confnorm, labels=classes, title="%s Confusion Matrix (SNR=%d)" % (name, snr))

    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print ("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)

# %%Save results to a pickle file for plotting later
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

# %%Plot accuracy curve
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("%s Classification Accuracy on RadioML 2016.10 Alpha" % (name))
plt.savefig("%s Classification Accuracy" % (name))
# plt.show()