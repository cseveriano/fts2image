import pandas as pd
import os
from pyFTS.partitioners import Grid, Entropy, Util as pUtil

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

#Normalize Data

# Save Min-Max for Denorm
min_raw = train_df.min()

max_raw = train_df.max()

# Perform Normalization
norm_df = normalize(train_df)

npartitions = 50
nlags = 4
steps = 1
fuzzy_sets = Grid.GridPartitioner(data=norm_df.values, npart=npartitions)
model = FuzzyImageCNN(fuzzy_sets, nlags=nlags, steps=steps)
model.fit(norm_df.values,epochs=5, batch_size=64)

#model.predict()


