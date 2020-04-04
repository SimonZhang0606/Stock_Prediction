"""
Created on April 4 2020
@author: SimonZhang0606
Predicts the next day (closing) stock prices for APPL data using LSTM,
"""

#modules that i can think of for now
import math 
import pandas_datareader as web
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')


#Reading in the data
df = web.DataReader('AAPL', data_source = 'yahoo', start = "2012-01-01", end = "2019-12-19" )
print(df)


#showing the dashboard
plt.figure(figsize = (16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

#Create a new dataframe with only 'Close column'
data = df.filter(['Close'])

#conver the dataframe to a numpy array 
dataset = data.values


#Scale the data 
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)


#create the training dataset 

#Create the scaled training data set
train_data = scaled_data[0:training_data_len, :]

#split the data into x_train and y_train data sets 
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60: i,0])
    y_train.append(train_data[i,0])

    if i <= 60:
        print(x_train)
        print(y_train)
        print()

    #convert x_train and y_train to numpy arrays 

x_train, y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

x_train.shape()

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape(x_train.shape[1],1)))
model.add(LSTM(50,return_sequence = False ))
model.add(Dense(25))
model.add(Dense(1))


#compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#train the model 
model.fit(x_train, y_train, batch_size = 1, epochs = 1)


#create the testing data set 
#create an array containing scaled values from index 1543 - 2003
test_data = scaled_data[training_data_len - 60: , :]

#create the data sets x_test and y_test 
x_test = []

y_test = dataset[training_data_len:, :]

for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i])

#Convert the data to a numpy array 

x_test = np.array(x_test)

#Reshape the data 
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Get the model's predicted price values
predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)

rmse = np.sqrt(np.mean( predictions - y_test ) **2)
