import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv("Stock_Price_Train.csv")
test=pd.read_csv("Stock_Price_Test.csv")

open_train=train["Open"].values

open_train=((open_train-np.min(open_train))\
    /(np.max(open_train)-np.min(open_train))).reshape(-1,1)

plt.plot(open_train)

x_train=[]
x_test=[]

for i in range(50,open_train.shape[0]):
    x_train.append(open_train[i-50:i,0])
    x_test.append(open_train[i,0])
x_train,x_test=np.array(x_train),np.array(x_test)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#%%

from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Dropout

model=Sequential()

model.add(SimpleRNN(units=50,activation="tanh",return_sequences=True,\
            input_shape=(x_train.shape[1],1)))
model.add(Dropout(0,2))

model.add(SimpleRNN(units=50,activation="tanh",return_sequences=True,))
model.add(Dropout(0,2))

model.add(SimpleRNN(units=50,activation="tanh",return_sequences=True,))
model.add(Dropout(0,2))

model.add(SimpleRNN(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer="adam",loss="mean_squared_error")

model.fit(x_train,x_test,epochs=100 ,batch_size=32)