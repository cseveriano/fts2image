import pandas as pd
import numpy as np
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval
import traceback

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



###### CNN FUNCTIONS ###########
def fuzzy_cnn_forecast(train_df, test_df, params):
    model = OneDimensionalConvolutionCNN.OneDimensionalConvolutionCNN(input, output,  nlags=params['order'], steps=1,
                                                                      conv_layers=params['conv_layers'],
                                                                      filters=params['filters'],
                                                                      kernel_size=params['kernel_size'],
                                                                      pooling_size=params['pooling_size'],
                                                                      strides=params['strides'],
                                                                      dense_layer_neurons=params['dense_layer_neurons'])

    model.fit(train_df, batch_size=params['batch_size'], epochs=params['epochs'])

    forecast = model.predict(test_df)

    return forecast


def cnn_objective(params):
    print(params)
    try:
        forecast = fuzzy_cnn_forecast(norm_train_df, norm_validation_df, params)
        forecast = denormalize(forecast, df[output])
        rmse = calculate_rmse(validation_df[output], forecast, params['order'], 1)
    except Exception:
        traceback.print_exc()
        rmse = 1000

    return {'loss': rmse, 'status': STATUS_OK}


###### OPTIMIZATION ROUTINES ###########
space = {'order': hp.choice('order', [48,96,144]),
        'epochs': hp.choice('epochs', [100,200]),
        'batch_size': hp.choice('batch_size', [32,100,200]),
        'conv_layers' : hp.choice('conv_layers', list(np.arange(1,3))),
        'filters': hp.choice('filters',  [2, 4, 8, 16]),
        'kernel_size': hp.choice('kernel_size', list(np.arange(2,4))),
        'pooling_size': hp.choice('pooling_size', list(np.arange(2,4))),
        'strides': hp.choice('strides', [1,2]),
        'dense_layer_neurons': hp.choice('dense_layer_neurons', [[8], [32, 8], [64, 32, 8]])}


# trials = pickle.load(open("tuning_results.pkl", "rb"))
# best = pickle.load(open("best_result.pkl", "rb"))

trials = Trials()
best = fmin(cnn_objective, space, algo=tpe.suggest, max_evals =500, trials=trials)
print('best: ')
print(space_eval(space, best))
