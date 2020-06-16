import tensorflow as tf
import numpy as np

#Implementation of a neural network without keras
class TensorflowMnistModel :
    def __init__(self,_n_input, _n_output, _epoch, _batch_size, _lr):
        self.n_input = _n_input
        self.n_output = _n_output
        self.epoch = _epoch
        self.batch_size = _batch_size
        self.lr = _lr


    def buildModel(self, n_neurons_1):
        X = tf.keras.backend.placeholder(tf.float32, [None, self.n_input])
        Y = tf.keras.backend.placeholder(tf.float32, {None, self.n_output})

        init1 = tf.initializers.TruncatedNormal((self.n_input, n_neurons_1), stddev= 2 / np.sqrt(self.n_input+n_neurons_1))
        init2 = tf.initializers.TruncatedNormal((n_neurons_1, self.n_output), stddev= 2 / np.sqrt(self.n_output+n_neurons_1))
        w1 = tf.Variable(init1)
        w2 = tf.Variable(init2)
        b1 = tf.Variable(np.zeros(n_neurons_1, dtype="float32"))
        b2 = tf.Variable(np.zeros(self.n_output, dtype="float32"))

        z1 = tf.matmul(X,w1) + b1
        h1 = tf.nn.relu(z1)
        z2 = tf.matmul(h1,w2) + b2
        Y = tf.nn.relu(z2)


#Implementation of the same neural network with keras

class KerasMnistModel :
    def __init__(self,_n_input, _n_output, _epoch, _batch_size, _lr):
        self.n_input = _n_input
        self.n_output = _n_output
        self.epoch = _epoch
        self.batch_size = _batch_size
        self.lr = _lr