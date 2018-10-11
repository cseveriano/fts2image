import pandas as pd
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt

from fts2image import FuzzyImageCNN

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]


#Set target and input variables
target_station = 'DHHL_3'

df = pd.read_pickle(os.path.join(os.getcwd(), "../data/oahu/df_oahu.pkl"))


# Get data form the interval of interest
train_interval = ((df.index >= '2010-06') & (df.index < '2010-07'))
test_interval = ((df.index >= '2010-07') & (df.index < '2010-08'))

train_df = df[target_station].loc[train_interval]
test_df = df[target_station].loc[test_interval]

#Normalize Data

# Save Min-Max for Denorm
min_raw = train_df.min()

max_raw = train_df.max()

# Perform Normalization
train_norm_df = normalize(train_df)
test_norm_df = normalize(test_df)

_conv_layers = 2
_dense_layer_neurons = 1024
_dense_layers = 3
_epochs = 30
_filters = 8
_kernel_size = 2
_npartitions = 50
_order = 8
_pooling_size = 2

fuzzy_sets = Grid.GridPartitioner(data=train_df.values, npart=_npartitions).sets
model = FuzzyImageCNN.FuzzyImageCNN(fuzzy_sets, nlags=_order, steps=1,
                                    conv_layers=_conv_layers, dense_layers=_dense_layers,
                                    dense_layer_neurons=_dense_layer_neurons, filters=_filters,
                                    kernel_size=_kernel_size, pooling_size=_pooling_size)
model.fit(train_df, epochs=_epochs, batch_size=64)

forecast = model.predict(test_norm_df)

final_forecast = denormalize(forecast, min_raw, max_raw)

plt.figure()
plt.plot(test_df.iloc[1:600].values)
plt.plot(final_forecast[:599])
plt.show()
