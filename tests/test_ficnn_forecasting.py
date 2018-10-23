import pandas as pd
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from fts2image import FuzzyImageCNN

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test.iloc[(order):], forecast[:-step]))
    print("RMSE : "+str(rmse))
    return rmse

def split_data(df, interval):
    sample_df = df.loc[interval]

    week = (sample_df.index.day - 1) // 7 + 1

    # PARA OS TESTES:
    # 2 SEMANAS PARA TREINAMENTO
    train_df = sample_df.loc[week <= 2]

    # 1 SEMANA PARA VALIDACAO
    validation_df = sample_df.loc[week == 3]

    # 1 SEMANA PARA TESTES
    test_df = sample_df.loc[week > 3]

    return (train_df, validation_df, test_df)

#Set target and input variables
target_station = 'DHHL_3'

df = pd.read_pickle(os.path.join(os.getcwd(), "../data/oahu/df_oahu.pkl"))

#Normalize Data

# Save Min-Max for Denorm
min_raw = df[target_station].min()

max_raw = df[target_station].max()

# Perform Normalization
norm_df = normalize(df)

# Split data
interval = ((df.index >= '2010-06') & (df.index < '2010-07'))
#interval = ((df.index >= '2010-11') & (df.index <= '2010-12'))

(train_df, validation_df, test_df) = split_data(df, interval)
(norm_train_df, norm_validation_df, norm_test_df) = split_data(norm_df, interval)


def fuzzy_cnn_forecast(train_df, test_df):
    _conv_layers = 2
    _dense_layer_neurons = 1024
    _dense_layers = 3
    _epochs = 30
    _filters = 8
    _kernel_size = 2
    _npartitions = 50
    _order = 8
    _pooling_size = 2
    _dropout = 0

    fuzzy_sets = Grid.GridPartitioner(data=train_df.values, npart=_npartitions).sets
    model = FuzzyImageCNN.FuzzyImageCNN(fuzzy_sets, nlags=_order, steps=1,
                                        conv_layers=_conv_layers, dense_layers=_dense_layers,
                                        dense_layer_neurons=_dense_layer_neurons, filters=_filters,
                                        kernel_size=_kernel_size, pooling_size=_pooling_size, dropout=_dropout)
    model.fit(train_df, epochs=_epochs)

    forecast = model.predict(test_df)

    return forecast

steps = 1

forecast = fuzzy_cnn_forecast(norm_train_df[target_station], norm_test_df[target_station])
forecast = denormalize(forecast, min_raw, max_raw)

_order = 8

forecast.append(0) ## para manter o mesmo tamanho dos demais
rmse = calculate_rmse(test_df[target_station], forecast, _order, steps)
print("RMSE: ", rmse)

plt.figure()
plt.plot(test_df.iloc[_order:600].values)
plt.plot(forecast[:(600+_order)])
plt.show()
