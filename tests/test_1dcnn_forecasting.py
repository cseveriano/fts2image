import pandas as pd
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from fts2image import OneDimensionalConvolutionCNN

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm_data, original_data):
    min = original_data.min()

    max = original_data.max()

    return [(n * (max-min)) + min for n in norm_data]

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

multichannel = ['WTG01_Speed', 'WTG01_Power']
output = 'WTG01_Speed'

neighbor_stations_90 = ['WTG01_Speed','WTG02_Speed','WTG03_Speed','WTG05_Speed','WTG06_Speed']

one_station = ['WTG01_Speed']

all_stations = ['WTG01_Speed', 'WTG02_Speed', 'WTG03_Speed', 'WTG04_Speed', 'WTG05_Speed',
'WTG06_Speed', 'WTG07_Speed', 'WTG08_Speed', 'WTG09_Speed', 'WTG10_Speed',
'WTG11_Speed', 'WTG12_Speed', 'WTG13_Speed', 'WTG24_Speed', 'WTG25_Speed']

#input = neighbor_stations_90
input = neighbor_stations_90


df = pd.read_pickle(os.path.join(os.getcwd(), "../data/wind/df_wind_total.pkl"))

#Normalize Data

# Perform Normalization
norm_df = normalize(df)

# Split data
interval = ((df.index >= '2017-05') & (df.index <= '2017-06'))

(train_df, validation_df, test_df) = split_data(df, interval)
(norm_train_df, norm_validation_df, norm_test_df) = split_data(norm_df, interval)

#{'batch_size': 32, 'conv_layers': 1, 'dense_layer_neurons': (32, 8), 'epochs': 200, 'filters': 8, 'kernel_size': 3, 'order': 48, 'pooling_size': 2, 'strides': 1}


def fuzzy_cnn_forecast(train_df, test_df):
    _epochs = 200
    _batch_size = 32
    _conv_layers = 1
    _filters = 8
    _kernel_size = 3
    _pooling_size = 2
    _strides = 1
    _dense_layer_neurons = [32, 8]

    model = OneDimensionalConvolutionCNN.OneDimensionalConvolutionCNN(input, output, nlags=_order, steps=1,
                                                                      conv_layers=_conv_layers,
                                                                      filters=_filters,
                                                                      kernel_size=_kernel_size,
                                                                      pooling_size=_pooling_size,
                                                                      strides=_strides,
                                                                      dense_layer_neurons=_dense_layer_neurons
                                                                      )
    model.fit(train_df, batch_size=_batch_size, epochs=_epochs, verbose = False)

    forecast = model.predict(test_df)

    return forecast

steps = 1
_order = 48

forecast = fuzzy_cnn_forecast(norm_train_df, norm_validation_df)
forecast = denormalize(forecast, df[output])

rmse = calculate_rmse(validation_df[output], forecast, _order, steps)
print("RMSE: ", rmse)

plt.figure()
plt.plot(validation_df[output].iloc[_order:600].values)
plt.plot(forecast[:(600-_order)])
plt.show()
