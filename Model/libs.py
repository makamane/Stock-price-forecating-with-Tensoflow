# This impor all necessary moduls we will need
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from loadingData import loadData
# Loading data from 2012-01-01 to 2019-12-17
df = loadData('2012-01-01', '2019-12-17')

# # loading data tUserWarningse
# df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
# # Show teh data
print(df.shape)

# # Visualize the closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History - Data as it is.')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

# Create a new dataframe with only the close column
data = df.filter(['Close'])
# Conver the dataframe to a nompy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)
training_data_len

# Scale the data to be between 0-1 A.K.A normalization  data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training dataset
#Create the scaled traing dataset
train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train dataset
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
#    if i<=61:
#        print(x_train)
#        print(y_train)

#Convert the  x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))
x_train.shape

# Buils the LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(tf.keras.layers.LSTM(50, return_sequences=False))
model.add(tf.keras.layers.Dense(25))
model.add(tf.keras.layers.Dense(1))


# Compiling the model

model.compile(optimizer='adam', loss='mean_squared_error')

# 'mean_squared_error', 'sparse_categorical_crossentropy'
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5)

# create the test dataset
#create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60:]
#crreate the dataset x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

# convert the data to numpy array
x_test = np.array(x_test)
x_test.shape

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape

# Get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean square error (RMSE)
rmse =np.sqrt(np.mean(predictions - y_test)**2)
print('Model RMSE: ', rmse)

# accScore = model.evaluate(x_test,  y_test, verbose=2)
# print("Accuracy Score: ", accScore)
# print("Model Evaluation:",model.evaluate(x_test, y_test))

# #Plot the data
# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid['Predictions'] = predictions
#
# # Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=10)
# plt.ylabel('Close Price USD ($)', fontsize=10)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'Predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

#Get the quote
apple_quote = loadData(startDate ='2012-01-01', endDate='2019-12-17')
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#Get the last 60 day closing price values conver the dataframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 nad 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Conver X_test dataset to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print('18 Dec 2019 prediction:',pred_price)

#Get the quote
apple_quote2 = loadData(startDate='2019-12-18', endDate='2019-12-18')
print('18 Dec 2019 Actual Amount:', apple_quote2['Close'])
