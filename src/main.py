import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

N_SEQ = 0
TIMESTEPS = 0
N_FEATURES = 0

print(tf.__version__)

