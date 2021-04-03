import tensorflow as tf
import pandas as pd
import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


n_steps_in = 60 # one sequence will have 60 elements of data
n_steps_out = 3 # goal is to predict 5 elements of data 

def is_profit(actual, future):
	if float(future) > float(actual):
		return 1
	return 0

def preprocess_df(df):
	df = df.drop(columns='future')
	df.fillna(method="ffill", inplace=True)
	df[['close']] = df[['close']].apply(lambda x:(x-x.min()) / (x.max()-x.min()))
	sequence_list = list()
	sequence = deque(maxlen=n_steps_in)
	for row in df.values:
		sequence.append([col for col in row[:-1]])
		if len(sequence) == n_steps_in:
			sequence_list.append([sequence, row[-1]])
	profit = list(); loss = list()
	for seq, target in sequence_list:
		if target == 1:
			profit.append([seq, target])
		elif target == 0:
			loss.append([seq, target])
	sequence_list = profit + loss
	random.shuffle(sequence_list)
	x = list(); y = list()
	for seq, target in sequence_list:
		x.append(np.array(seq))
		y.append(target)
	return np.array(x), np.array(y)


df = pd.read_csv("./src/data/BTC-USD.csv")
df.set_index('time', inplace=True)
df.sort_index(inplace=True)
df.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)

# create future column, create target/label column
df['future'] = df['close'].shift(-n_steps_out)
df['target'] = list(map(is_profit, df['close'], df['future']))

# split data into train, test sections (70/30) split
timestamp = df.iloc[-int(df.shape[0] * 0.3)]
train_df = df.loc[df.index < timestamp.name]
test_df = df.loc[df.index >= timestamp.name]

# preprocess_df
	# normalize and scale columns
	# balance data (shuffle and balance profits and losses
	# split sequences from labels
x_train, y_train = preprocess_df(train_df)
x_test, y_test = preprocess_df(test_df)

x_train = x_train.reshape(x_train.shape[0], n_steps_in, 1)
x_test = x_test.reshape(x_test.shape[0], n_steps_in, 1)

# pass data through neural network.
model = Sequential()

model.add(LSTM(64, activation="tanh"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#model.add(LSTM(1, activation="sigmoid"))
#model.add(Dropout(0.2))
#model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_steps_out))

opt = Adam(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=False)
guess = model.predict(x_test, verbose=1)
print('prediction: ', guess)
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

