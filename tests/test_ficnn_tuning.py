import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

import traceback

import math
from sklearn.metrics import mean_squared_error
from pyFTS.partitioners import Grid, Entropy, Util as pUtil

from fts2image import FuzzyImageCNN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval


###### COMMON FUNCTIONS ###########

def normalize(df):
    mindf = df.min()
    maxdf = df.max()
    return (df-mindf)/(maxdf-mindf)

def denormalize(norm, _min, _max):
    return [(n * (_max-_min)) + _min for n in norm]


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

def calculate_rmse(test, forecast, order, step):
    rmse = math.sqrt(mean_squared_error(test.iloc[(order):], forecast[:-step]))
    print("RMSE : "+str(rmse))
    return rmse


###### CNN FUNCTIONS ###########
def fuzzy_cnn_forecast(train_df, test_df, params):
    fuzzy_sets = Grid.GridPartitioner(data=train_df.values, npart=params['npartitions']).sets
    model = FuzzyImageCNN.FuzzyImageCNN(fuzzy_sets, nlags=params['order'], steps=1,
            conv_layers = params['conv_layers'], dense_layers = params['dense_layers'],
            filters = params['filters'], kernel_size = params['kernel_size'],
            pooling_size = params['pooling_size'], dense_layer_neurons = params['dense_layer_neurons'])

    model.fit(train_df, epochs=params['epochs'])

    forecast = model.predict(test_df)

    return forecast


def cnn_objective(params):
    print(params)
    try:
        forecast = fuzzy_cnn_forecast(norm_train_df[target_station], norm_test_df[target_station], params)
        forecast = denormalize(forecast, min_raw, max_raw)
        forecast.append(0) ## para manter o mesmo tamanho dos demais
        rmse = calculate_rmse(test_df[target_station], forecast, params['order'], 1)
    except Exception:
        traceback.print_exc()
        rmse = 1000

    return {'loss': rmse, 'status': STATUS_OK}


###### LOAD DATA ###########


#Set target and input variables
target_station = 'DHHL_3'

df = pd.read_pickle(os.path.join(os.getcwd(), "../data/oahu/df_oahu.pkl"))

## Remove columns with many corrupted or missing values
df.drop(columns=['AP_1', 'AP_7'], inplace=True)

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


###### OPTIMIZATION ROUTINES ###########
space = {'npartitions': hp.choice('npartitions', [50, 100, 150, 200]),
        'order': hp.choice('order', list(np.arange(4,10))),
        'epochs': hp.choice('epochs', [30, 50, 100]),
        'conv_layers' : hp.choice('conv_layers', list(np.arange(2,6))),
        'dense_layers': hp.choice('dense_layers', list(np.arange(2, 6))),
        'filters': hp.choice('filters',  [2, 4, 8, 16, 32, 64]),
        'kernel_size': hp.choice('kernel_size', list(np.arange(2,5))),
        'pooling_size': hp.choice('pooling_size', list(np.arange(2, 5))),
        'dense_layer_neurons': hp.choice('dense_layer_neurons', [32, 64, 128, 256, 512, 768, 1024, 1280]),
        'dropout': hp.choice('dropout', list(np.arange(0.2, 0.5, 0.1)))}


trials = pickle.load(open("tuning_results.pkl", "rb"))
best = pickle.load(open("best_result.pkl", "rb"))

trials = Trials()
best = fmin(cnn_objective, space, algo=tpe.suggest, max_evals =500, trials=trials)
print('best: ')
print(space_eval(space, best))

pickle.dump(best, open("best_result.pkl", "wb"))
pickle.dump(trials, open("tuning_results.pkl", "wb"))

