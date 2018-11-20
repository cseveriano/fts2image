import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, concatenate

from scipy.ndimage import interpolation

class MultiChannelFuzzyImageCNN:

    def __init__(self, raw_channels, fuzzy_sets, output, nlags=1, steps=1,
                 conv_layers=1, dense_layers=1, filters=32, kernel_size=3, pooling_size=2, dense_layer_neurons=64, dropout=0):
        self.nlags = nlags
        self.steps = steps
        self.raw_channels = raw_channels
        self.fuzzysets = fuzzy_sets
        self.output = output
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.filters = filters
        self.kernel_size = [kernel_size,kernel_size]
        self.dense_layer_neurons = dense_layer_neurons
        self.pooling = (pooling_size, pooling_size)
        self.dropout = dropout

    def convert_to_one_hot_image(self, sequence):
        from keras.utils import to_categorical

        sorted_sequence = sorted(sequence)
        ind_sequence = [sorted_sequence.index(k) for k in sequence]
        one_hot_sequence = []
        for ind in ind_sequence:
            seq = [0] * self.nlags
            seq[ind] = 1
            one_hot_sequence.append(seq)

        return one_hot_sequence

    def series_to_supervised(self, data, n_vars=1, lags=1, steps=1, dropnan=True):

        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(lags, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, steps):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


    def design_network(self):
        # insert proper configs
        self.model = Sequential()

        for i in np.arange(self.conv_layers):
            self.model.add(Conv2D(self.filters * (i+1), self.kernel_size, padding="same", activation='relu', input_shape=(self.nlags, self.nlags, len(self.raw_channels)+1)))
            self.model.add(MaxPooling2D(self.pooling, padding="same"))

        self.model.add(Flatten())

        if self.dropout > 0:
            self.model.add(Dropout(self.dropout))

        for i in np.arange(self.dense_layers):
            self.model.add(Dense(self.dense_layer_neurons, activation='relu'))

        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

    # def design_network(self):
    #     # insert proper configs
    #     self.model = Sequential()
    #
    #     self.model.add(Conv2D(32, (2,2), padding="same", activation='relu', input_shape=(self.nlags, self.nlags, len(self.raw_channels)+1)))
    #     self.model.add(MaxPooling2D((2,2), padding="same", strides=2))
    #
    #     self.model.add(Conv2D(64, (2,2), padding="same", activation='relu', input_shape=(self.nlags, self.nlags, len(self.raw_channels)+1)))
    #     self.model.add(MaxPooling2D((2,2), padding="same", strides=2))
    #
    #     self.model.add(Flatten())
    #     self.model.add(Dense(4096, activation='relu'))
    #     self.model.add(Dense(1024, activation='relu'))
    #
    #    # self.model.add(Dropout(0.4))
    #
    #     self.model.add(Dense(50, activation='relu'))
    #     self.model.add(Dense(40, activation='relu'))
    #     self.model.add(Dense(5, activation='relu'))
    #
    #     self.model.add(Dense(1, activation='linear'))
    #     self.model.compile(loss='mse', optimizer='adam')

    @staticmethod
    def plotImage(image):
        plt.imshow(image, cmap="gray")
        plt.show()

    def fit(self, train_data, epochs=20, plot_images=False):

        X_images = self.create_images(train_data)

        if plot_images:
            for image in X_images:
                MultiChannelFuzzyImageCNN.plotImage(image[0])


        # Create y data
        y = self.create_output(train_data)

        self.design_network()
        self.model.fit(X_images, y, epochs)

    def create_output(self, data):
        sup_data = self.series_to_supervised(data[self.output], lags=self.nlags, steps=self.steps)
        y = np.array(sup_data.iloc[:, -self.steps].values)
        return y

    def create_images(self, data):
        X_images = []
        # Create X raw channels data
        for channel in self.raw_channels:
            sup_data = self.series_to_supervised(data[channel], lags=self.nlags, steps=self.steps)

            X = np.array(sup_data.iloc[:, :self.nlags].values)

            channel_matrix = []

            for sample in X:
                channel_matrix.append(self.convert_to_one_hot_image(sample))

            X_images.append(channel_matrix)
        # Create fuzzy image
        fuzzy_channel_matrix = []
        sup_data = self.series_to_supervised(data[self.raw_channels[0]], lags=self.nlags, steps=self.steps)
        X = np.array(sup_data.iloc[:, :self.nlags].values)
        for sample in X:
            fuzzy_sequence = []
            for lag in sample:
                memberships = [self.fuzzysets[fs].membership(lag) for fs in self.fuzzysets]
                fuzzy_sequence.append(np.argmax(memberships))
            fuzzy_channel_matrix.append(self.convert_to_one_hot_image(fuzzy_sequence))
        X_images.append(fuzzy_channel_matrix)
        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(data) - (self.nlags + (self.steps - 1)), self.nlags, self.nlags,
                                    len(self.raw_channels) + 1)
        return X_images

    def predict(self, test_data):
        X_images = self.create_images(test_data)
        return self.model.predict(X_images)
