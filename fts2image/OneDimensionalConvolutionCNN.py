import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv1D, Flatten, Dense, MaxPooling1D, LSTM

from scipy.ndimage import interpolation

class OneDimensionalConvolutionCNN:

    def __init__(self, inputs, output, nlags=1, steps=1):
        self.nlags = nlags
        self.steps = steps
        self.inputs = inputs
        self.output = output


    def design_network(self):

        self.model = Sequential()
        self.model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.nlags, len(self.inputs))))
        self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(Flatten())
        # self.model.add(Dense(50, activation='relu'))
        self.model.add(LSTM(50, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

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

    def fit(self, train_data, batch_size, epochs):

        X, y = self.create_supervised_dataset(train_data)

        self.design_network()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

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
