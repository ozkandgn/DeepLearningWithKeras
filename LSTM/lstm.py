import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

data=pd.read_csv("airline_data.csv")
data.drop((data.shape[0]-1),axis=0,inplace=True)

plt.plot(data.iloc[:,1])
plt.show()

data.drop(("Month"),axis=1,inplace=True)
data=data.values.reshape(-1,1)
data=data.astype("float32")

scaler=MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(data)

train=int(data.shape[0]/2)
test=int(data.shape[0]-train)

train=data[0:train,:]
test=data[test:data.shape[0],:]

temp_x=[]
temp_y=[]

time_stemp=12

for  i in range(train.shape[0]-time_stemp-1):
    temp=train[i:(i+time_stemp),0]
    temp_x.append(temp)
    temp_y.append(train[i+time_stemp,0])

train_x=np.array(temp_x)
train_y=np.array(temp_y)

for  i in range(test.shape[0]-time_stemp-1):
    temp=test[i:(i+time_stemp),0]
    temp_x.append(temp)
    temp_y.append(test[i+time_stemp,0])

test_x=np.array(temp_x)
test_y=np.array(temp_y)

train_x=np.reshape(train_x,(train_x.shape[0],1,train_x.shape[1]))
test_x=np.reshape(test_x,(test_x.shape[0],1,test_x.shape[1]))

model=Sequential()
model.add(LSTM(10,input_shape=(1,time_stemp)))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")
model.fit(train_x,train_y,epochs=100,batch_size=10)

train_predict=model.predict(train_x)
test_predict=model.predict(test_x)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)
train_y=scaler.inverse_transform([train_y])
test_y=scaler.inverse_transform([test_y])

plt.plot(train_predict,color="red",alpha=0.5)
plt.plot(test_predict,color="red",alpha=0.5)
plt.plot(train_y.T,color="black",alpha=0.5)
plt.plot(test_y.T,color="black",alpha=0.5)