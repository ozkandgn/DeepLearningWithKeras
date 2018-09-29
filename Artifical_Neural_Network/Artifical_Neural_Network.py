from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data=pd.read_csv('voice.csv')

male=data[data.label == "male"]
female=data[data.label == "female"]

y=pd.DataFrame([(1 if i=="male" else 0) for i in pd.DataFrame(data["label"]).values])

x=data.drop(["label"],axis=1)

x=(x-np.min(x))/(np.max(x)-np.min(x))

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8,kernel_initializer='uniform',\
                    activation='relu',input_dim=x.shape[1]))
    classifier.add(Dense(units= 4,kernel_initializer='uniform',\
                         activation='relu'))
    classifier.add(Dense(units= 1,kernel_initializer='uniform',\
                         activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',\
                       metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,epochs=50)
accuracies=cross_val_score(estimator=classifier,X=x,y=y,cv=2)
print("Accuracy=",accuracies.mean())