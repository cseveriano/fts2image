import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

class FuzzyImageCNN:

    def __init__(self, fuzzysets, nlags=1, steps=1):
        self.nlags = nlags
        self.steps = steps
        self.fuzzysets = fuzzysets

    def convert2image(self, sequence):
        image = np.zeros(shape=(len(sequence), len(self.fuzzysets)))

        for i, lag in enumerate(sequence):
            for j, fs in  enumerate(self.fuzzysets):
                image[i, j] = self.fuzzysets[fs].membership(lag)

        return image

    # convert series to supervised learning
    def series_to_supervised(self, series, dropnan=True):
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

    def design_network(self):
        #insert proper configs
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(self.nlags, len(self.fuzzysets), 1)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
  #      self.model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse', optimizer='adam')

    def fit(self, train_data, epochs=5, batch_size=64):
        sup_data = self.series_to_supervised(train_data)

        X = np.array(sup_data.iloc[:, :self.nlags].values)
        y = np.array(sup_data.iloc[:,-self.steps].values)

        X_images = []

        for sample in X:
            X_images.append(self.convert2image(sample))

        # reshape input values according to network architecture
        X_images = np.array(X_images)
        X_images = X_images.reshape(len(X_images), self.nlags, len(self.fuzzysets), 1)


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
        X_images = X_images.reshape(len(X_images), self.nlags, len(self.fuzzysets), 1)

        return self.model.predict(X_images)
