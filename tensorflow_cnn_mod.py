import tensorflow as tf
import numpy as np
import math
import cPickle

Xd = cPickle.load(open("RML2016.10a_dict.dat", 'rb'))
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
modul = ['QAM16', 'QAM64']
for mod in mods:
    if mod in modul:
        for snr in snrs:
            with open("dataset.txt", "w") as f:
                for s in Xd[(mod, snr)]:
                    f.write(str(s) + str(mod) + "\n")
            f.close()
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))

X = np.vstack(X)
np.random.seed(1263)
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

# Training Parameters
initial_learning_rate = 0.001
maxEpochs = 1000
batch_size = 150

#network parameters
num_input = 2*128
num_classes = len(mods)
keep_prob = 0.5

initializer = tf.contrib.layers.xavier_initializer()


def conv_layer(x, filt_shape, name="conv",strides = [1,1,1,1],padding = "SAME"):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=filt_shape,initializer=initializer)
        bias = tf.Variable(tf.constant(0.1, shape=[filt_shape[-1]]), name = 'bias_'+name)
        conv = tf.add(tf.nn.conv2d(x, Weights, strides=strides, padding=padding), bias)
        return conv


def fully_connected_layer(x, shape, keep_prob, dropconnect=False, name ="Fully_Connected"):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=shape,initializer=initializer)
        bias = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name = 'bias_'+name)
        fc = tf.matmul(x, Weights) + bias
        fc = tf.nn.relu(fc)
        if dropconnect :
            return tf.nn.dropout(fc, keep_prob)*keep_prob # for regularization
        else:
            return tf.nn.dropout(fc, keep_prob)


def output_layer(x, shape, name='output'):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=shape,initializer=initializer)
        bias = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name = 'bias_'+name)
        return tf.matmul(x, Weights) + bias


def flatten(x,shape,name='Flatten'):
    with tf.name_scope(name) as scope:
        flattened = tf.reshape(x, [-1, shape])
        return flattened


def batch_norm_layer(x,train,name='batch_norm'):

    with tf.name_scope(name) as scope:
        conv = tf.contrib.layers.batch_norm(x,is_training=train,updates_collections=None)
        return tf.nn.relu(conv)


with tf.variable_scope("placeholders"):
    inputs = tf.placeholder(tf.float32, shape=[None, num_input], name="x-data")
    labels = tf.placeholder(tf.int64, shape=[None], name="y-labels")
    dropout = tf.placeholder(tf.float32)
    train = tf.placeholder(tf.bool)


def cnn_lstm(x, num_class, keep_prob, train):

    x = tf.reshape(x, shape=[-1, num_input, 1, 1],name = "Reshape_data_1")

    conv1 = conv_layer(x, [7, 1, 1, 128],name="conv1")
    norm1 = batch_norm_layer(conv1, train, name = "batch_norm_conv1")

    conv2 = conv_layer(norm1, [5, 1, 128, 50], name = "conv2")
    norm2 = batch_norm_layer(conv2, train, name = "batch_norm_conv2")

    conv3 = conv_layer(norm1, [5, 1, 50, 50], name="conv3")
    norm3 = batch_norm_layer(conv3, train, name="batch_norm_conv3")

    norm2_f = flatten(norm2, num_input*50)

    fc = fully_connected_layer(norm2_f ,
                                [num_input*50, 1000],
                                keep_prob,
                                dropconnect=False,
                                name ="Fully_Connected1")

    out = output_layer(fc, [1000, num_class], name ="out_layer")

    return out


logits = cnn_lstm(inputs,  num_classes, keep_prob, train)
prediction = tf.argmax(tf.nn.softmax(logits), axis=1)

with tf.variable_scope('Loss'):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name="SoftmaxLoss")
    loss_op = tf.reduce_mean(losses)

with tf.variable_scope('Metrics'):
    acc,acc_op = tf.metrics.accuracy(labels=labels ,predictions=prediction,name="accuracy")
    acc2 = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), "float"))
    rec,rec_op = tf.metrics.recall(labels=labels ,predictions=prediction,name="recall")
    prec,prec_op = tf.metrics.precision(labels=labels ,predictions=prediction,name="precision")

with tf.variable_scope('Optimizer'):
    global_step = tf.Variable(1, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               1000, 0.8,
                                               staircase=True)
    adam = tf.train.AdamOptimizer(learning_rate)
    train_op = adam.minimize(loss_op, name="train_op",global_step=global_step)

# Launch the graph
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("Starting Training.")
    epoch = 0
    numLoop = int(math.ceil(X_train.shape[0] / batch_size))
    while True:
        for num in range(numLoop):
            batch_x = X_train[num *
                           batch_size:min((num + 1) * batch_size,
                                          X_train.shape[0])]
            batch_y = Y_train[num *
                           batch_size:min((num + 1) * batch_size,
                                          X_train.shape[0])]
            _, loss, step, lr = sess.run([train_op,
                                          loss_op,
                                          global_step,
                                          learning_rate],
                                         feed_dict={
                                                    inputs: batch_x,
                                                    labels: batch_y,
                                                    train:True,
                                                    dropout:keep_prob
                                                })
            if step % 20 == 0:
                accc, recc, precc, accc2 = sess.run([acc_op,rec_op,prec_op,acc2], feed_dict={
                    inputs: X_test,
                    labels: Y_test,
                    train:False,
                    dropout:1.
                })
                print('epoch: %d - iter: %d - lr: %f - Trainloss: %f - TestAccuracy: %f - acc2: %f - TestRecall: %f - TestPrecision: %f'
                      % (epoch,step,lr,loss,accc,accc2,recc,precc))

        epoch +=1
        if epoch%maxEpochs == 0:
            break


    print("Optimization Finished!")
    saver.save(sess, "./model.ckpt")
    print("Model has been saved in model.ckpt")
