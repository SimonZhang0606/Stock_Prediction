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


