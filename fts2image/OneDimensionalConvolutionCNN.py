import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, LSTM, Dropout, BatchNormalization, LeakyReLU

from scipy.ndimage import interpolation

class OneDimensionalConvolutionCNN:

    def __init__(self, inputs, output, nlags=1, steps=1,
                 conv_layers=1, filters=32, kernel_size=3, pooling_size=2, strides=1, dense_layer_neurons=None,
                 dropout=0
                 ):
        self.nlags = nlags
        self.steps = steps
        self.inputs = inputs
        self.output = output
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_size = int(kernel_size)
        self.pooling_size = int(pooling_size)
        self.strides = int(strides)
        self.dense_layer_neurons = dense_layer_neurons


    def design_network(self):

        self.model = Sequential()

        for i in np.arange(self.conv_layers):
            self.model.add(Conv1D(self.filters * (i+1), self.kernel_size, strides=self.strides, activation='relu', input_shape=(self.nlags, len(self.inputs))))
            self.model.add(MaxPooling1D(self.pooling_size))

        self.model.add(Flatten())

        if self.dense_layer_neurons is not None:
            for neurons in self.dense_layer_neurons:
                self.model.add(Dense(neurons, activation='relu'))

        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')


    # split a multivariate sequence into samples
    def split_sequences(self, sequences):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + self.nlags
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def fit(self, train_data, batch_size, epochs, verbose = True):

        X, y = self.create_supervised_dataset(train_data)

        self.design_network()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def create_supervised_dataset(self, data):
        input_array = data[self.inputs].values
        output_array = data[self.output].values
        output_array = output_array.reshape((len(output_array), self.steps))
        dataset = np.hstack((input_array, output_array))
        X, y = self.split_sequences(dataset)
        return X, y

    def predict(self, test_data):
        X, y = self.create_supervised_dataset(test_data)
        return self.model.predict(X)
