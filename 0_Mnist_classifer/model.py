import tensorflow as tf
import numpy as np

#Implementation of a simple neural neural network with keras

class KerasMnistModel :
    def __init__(self,_input_shape, _output_shape):
        self.input_shape = _input_shape
        self.output_shape = _output_shape
        self.model = self.buildModel()

    def buildModel(self) :
        image_input = tf.keras.Input(shape= self.input_shape)
        x = tf.keras.layers.Flatten()(image_input)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.activations.sigmoid(x)
        x = tf.keras.layers.Dense(10)(x)
        x = tf.keras.activations.softmax(x)
        model = tf.keras.Model(inputs= image_input, outputs= x)
        return model
    
    def printModel(self):
        self.model.summary()

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer= optimizer, loss= loss, metrics= metrics)
    #Will be modify with generator
    def train(self,x,y, batch_size, epoch):
        self.model.fit(x,y,batch_size= batch_size, epochs= epoch)
    
    def save(self,path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model.load_model(path)
