import pandas as pd
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

from fts2image import MultiChannelFuzzyImageCNN

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

channel_1 = 'WTG01_Speed'
channel_2 = 'WTG01_Dir'
output = 'WTG01_Speed'

df = pd.read_pickle(os.path.join(os.getcwd(), "../data/wind/df_wind_total.pkl"))

#Normalize Data

# Perform Normalization
norm_df = normalize(df)

# Split data
interval = ((df.index >= '2017-05') & (df.index <= '2017-06'))

(train_df, validation_df, test_df) = split_data(df, interval)
(norm_train_df, norm_validation_df, norm_test_df) = split_data(norm_df, interval)


def fuzzy_cnn_forecast(train_df, test_df):
    _conv_layers = 3
    _dense_layer_neurons = 1280
    _dense_layers = 5
    _epochs = 100
    _filters = 8
    _kernel_size = 3
    _npartitions = 100
    _order = 100
    _pooling_size = 4
    _dropout = 0.30

    fuzzy_sets = Grid.GridPartitioner(data=train_df[channel_1].values, npart=_npartitions).sets

    model = MultiChannelFuzzyImageCNN.MultiChannelFuzzyImageCNN([channel_1, channel_2],fuzzy_sets, channel_1, nlags=_order, steps=1,
                                        conv_layers=_conv_layers, dense_layers=_dense_layers,
                                        dense_layer_neurons=_dense_layer_neurons, filters=_filters,
                                        kernel_size=_kernel_size, pooling_size=_pooling_size, dropout=_dropout)
    model.fit(train_df, epochs=_epochs, plot_images=False)

    forecast = model.predict(test_df)

    return forecast

steps = 1

forecast = fuzzy_cnn_forecast(norm_train_df, norm_test_df)
forecast = denormalize(forecast, df[output])

_order = 100

forecast.append(0) ## para manter o mesmo tamanho dos demais
rmse = calculate_rmse(test_df[output], forecast, _order, steps)
print("RMSE: ", rmse)

plt.figure()
plt.plot(test_df[output].iloc[_order:600].values)
plt.plot(forecast[:(600-_order)])
plt.show()
