import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, concatenate

from scipy.ndimage import interpolation

class FuzzyImageCNN:

    def __init__(self, fuzzysets, nlags=1, steps=1,
                 conv_layers=1, filters=32, kernel_size=3, pooling_size=2, dense_layer_neurons=64, dropout=0, debug=False):
        self.nlags = nlags
        self.steps = steps
        self.fuzzysets = fuzzysets
        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_size = [kernel_size,kernel_size]
        self.dense_layer_neurons = dense_layer_neurons
        self.pooling = (pooling_size, pooling_size)
        self.dropout = dropout
        self.debug = debug

    def convert2image(self, sequence):
        image = np.zeros(shape=(len(sequence), len(self.fuzzysets)))

        for i, lag in enumerate(sequence):
            for j, fs in  enumerate(self.fuzzysets):
                image[i, j] = self.fuzzysets[fs].membership(lag)

        return image

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
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
            self.model.add(Conv2D(self.filters * (i+1), self.kernel_size, activation='relu', input_shape=(self.nlags * self.nfeatures, len(self.fuzzysets), 1)))
            self.model.add(MaxPooling2D(self.pooling))

        self.model.add(Flatten())

        if self.dropout > 0:
            self.model.add(Dropout(self.dropout))

        for neurons in self.dense_layer_neurons:
            self.model.add(Dense(neurons, activation='relu'))

        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')


    @staticmethod
    def plotImage(image):
        plt.imshow(image, cmap="gray")
        plt.show()

    def fit(self, train_data, batch_size=10, epochs=1, plot_images=False):
        sup_data = self.series_to_supervised(train_data,  n_in=self.nlags, n_out=self.steps)
        self.nfeatures = train_data.shape[1]

        X = np.array(sup_data.iloc[:, :(self.nlags * self.nfeatures)].values)
        y = np.array(sup_data.iloc[:,-self.nfeatures].values)

        X_images = []

        for sample in X:
            X_images.append(self.convert2image(sample))

        if plot_images:
            for image in X_images:
                FuzzyImageCNN.plotImage(image)

        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(X_images), self.nlags * self.nfeatures, len(self.fuzzysets), 1)


        self.design_network()

        if not debug:
            _verbose = 0
        self.model.fit(X_images, y, batch_size=batch_size, epochs=epochs, verbose=_verbose)

    def predict(self, test_data):
        sup_data = self.series_to_supervised(test_data, n_in=self.nlags, n_out=self.steps)
        X = np.array(sup_data.iloc[:, :(self.nlags * self.nfeatures)].values)
        y = np.array(sup_data.iloc[:,-self.nfeatures].values)

        X_images = []

        for sample in X:
            X_images.append(self.convert2image(sample))

        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(X_images), self.nlags * self.nfeatures, len(self.fuzzysets), 1)

        return self.model.predict(X_images)
