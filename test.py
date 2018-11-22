import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from datetime import datetime

oktober = pd.read_csv('oktober2018.csv', delimiter=',+')

start = np.array(oktober.values[:,0],np.newaxis)
stop  = np.array(oktober.values[:,2],np.newaxis)

a = list(oktober.values[:,1])

day  = np.zeros_like(start)
hour = np.zeros_like(start)
print("begynner")
for i in range(len(start)):
    print(i)
    day[i] = int(a[i][8:10])%7
    hour[i] =int(a[i][11:13]) 
    #day[i] = datetime.strptime(oktober.values[i,1],"%Y-%m-%d %H:%M:%S %z").timetuple().tm_wday
    #hour[i] = datetime.strptime(oktober.values[i,1],"%Y-%m-%d %H:%M:%S %z").timetuple().tm_hour
print("ferri")

#%%
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(stop)
encoded_Y = encoder.transform(stop)
# convert integers to dummy variables (i.e. one hot encoded)
onehot_stop = np_utils.to_categorical(encoded_Y)

X = np.c_[start,day,hour]


X_train,X_test,y_train,y_test = train_test_split(X,onehot_stop,test_size = 0.2)

#datetime.strptime(oktober.values[0,1],"%Y-%m-%d %H:%M:%S %z")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128,activation="relu",input_dim=3))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(62,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(onehot_stop.shape[1],activation="softmax"))


model.compile(optimizer='adam',loss = "categorical_crossentropy",metrics = ["accuracy"])
model.fit(X_train,y_train,epochs=100,batch_size=10000,validation_data=[X_test,y_test])

print('accuracy', model.evaluate(X_test,y_test))