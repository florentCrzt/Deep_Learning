import numpy as np
import tensorflow as tf
from model import KerasMnistModel


#Download MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Number of training examples : ", len(x_train))
print("Number of testing examples : ", len(x_test))

#Normalize data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)

#x_train = x_train.as_type('float32')
#x_test = x_test.as_type('float32')

#x_train /= 255
#x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


#Prepare the model
mnist_model = KerasMnistModel(_input_shape=input_shape,_output_shape=(10,1))

mnist_model.printModel()

mnist_model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

mnist_model.train(x_train, y_train, batch_size= 8, epoch= 10)