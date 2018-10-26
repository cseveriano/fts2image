import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, concatenate

from scipy.ndimage import interpolation

class FuzzyImageCNN:

    def __init__(self, fuzzysets, nlags=1, steps=1,
                 conv_layers=1, dense_layers=1, filters=32, kernel_size=3, pooling_size=2, dense_layer_neurons=64, dropout=0):
        self.nlags = nlags
        self.steps = steps
        self.fuzzysets = fuzzysets
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.filters = filters
        self.kernel_size = [kernel_size,kernel_size]
        self.dense_layer_neurons = dense_layer_neurons
        self.pooling = (pooling_size, pooling_size)
        self.dropout = dropout
        self.image_dim = 256

    def convert2image(self, sequence):
        image = np.zeros(shape=(len(sequence), len(self.fuzzysets)))

        for i, lag in enumerate(sequence):
            for j, fs in  enumerate(self.fuzzysets):
                image[i, j] = self.fuzzysets[fs].membership(lag)

        # Resize image to be suitable as input for CNN
        resize_dim = self.image_dim
        scale = (float(resize_dim) / float(len(sequence)), float(resize_dim) / float(len(self.fuzzysets)))
        res_image = interpolation.zoom(image, scale, mode='nearest')

        return res_image

    # convert series to supervised learning
    def series_to_supervised(self, series, dropnan=True):
        # TODO: is this code adapted to multivariate series?
        n_vars = 1
        df = pd.DataFrame(series)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(self.nlags, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, self.steps):
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

    # def design_network(self):
    #     # TODO: design GoogleNet parameters
    #     # insert proper configs
    #     self.model = Sequential()
    #
    #     for i in np.arange(self.conv_layers):
    #         self.model.add(Conv2D(self.filters * (i+1), self.kernel_size, padding="same", activation='relu', input_shape=(self.image_dim, self.image_dim, 1)))
    #         self.model.add(MaxPooling2D(self.pooling))
    #
    #     self.model.add(Flatten())
    #
    #     if self.dropout > 0:
    #         self.model.add(Dropout(self.dropout))
    #
    #     for i in np.arange(self.dense_layers):
    #         self.model.add(Dense(self.dense_layer_neurons, activation='relu'))
    #
    #     self.model.add(Dense(1, activation='linear'))
    #     self.model.compile(loss='mse', optimizer='adam')


    def design_network(self):
        print("Custom V3")


        from keras import layers


        input_img = layers.Input(shape=(self.image_dim, self.image_dim, 1))

        seq = layers.Conv2D(self.filters, 1, activation='relu', strides=2, padding="same")(input_img)
        seq = layers.MaxPooling2D(16, 3, activation='relu', strides=2, padding="same")(seq)

        seq = layers.Conv2D(self.filters, self.kernel_size, activation='relu', strides=2, padding="same")(seq)
        seq = layers.Conv2D(self.filters * 2, self.kernel_size, activation='relu', strides=2, padding="same")(seq)
        seq = layers.MaxPooling2D(16, 3, activation='relu', strides=2, padding="same")(seq)


        branch_a = layers.Conv2D(self.filters, 1, activation='relu', strides=2)(seq)

        branch_b = layers.Conv2D(self.filters, 1, activation='relu')(seq)
        branch_b = layers.Conv2D(self.filters * 2, self.kernel_size, activation='relu', strides=2, padding="same")(branch_b)

        branch_c = layers.AveragePooling2D(3, strides=2, padding="same")(seq)
        branch_c = layers.Conv2D(self.filters, self.kernel_size, activation='relu', padding="same")(branch_c)

        branch_d = layers.Conv2D(self.filters, 1, activation='relu')(seq)
        branch_d = layers.Conv2D(self.filters * 2, self.kernel_size, activation='relu', padding="same")(branch_d)
        branch_d = layers.Conv2D(self.filters * 2, self.kernel_size, activation='relu', strides=2, padding="same")(branch_d)

        output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

        output = Flatten()(output)

        for i in np.arange(self.dense_layers):
            output = Dense(self.dense_layer_neurons, activation='relu')(output)

        out = Dense(1, activation='linear')(output)

        self.model = Model(inputs=input_img, outputs=out)
        self.model.compile(loss='mse', optimizer='adam')


    @staticmethod
    def plotImage(image):
        plt.imshow(image, cmap="gray")
        plt.show()

    def fit(self, train_data, epochs=5, plot_images=False):
        sup_data = self.series_to_supervised(train_data)

        X = np.array(sup_data.iloc[:, :self.nlags].values)
        y = np.array(sup_data.iloc[:,-self.steps].values)

        X_images = []

        for sample in X:
            X_images.append(self.convert2image(sample))

        if plot_images:
            for image in X_images:
                FuzzyImageCNN.plotImage(image)

        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(X_images), self.image_dim, self.image_dim, 1)


        self.design_network()
        self.model.fit(X_images, y, epochs)

    def predict(self, test_data):
        sup_data = self.series_to_supervised(test_data)
        X = np.array(sup_data.iloc[:, :self.nlags].values)
        y = np.array(sup_data.iloc[:,-self.steps].values)

        X_images = []

        for sample in X:
            X_images.append(self.convert2image(sample))

        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(X_images), self.image_dim, self.image_dim, 1)

        return self.model.predict(X_images)
