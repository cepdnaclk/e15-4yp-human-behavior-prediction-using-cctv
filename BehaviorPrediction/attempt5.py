
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


path_train = 'train.csv'
path_test = 'test.csv'

train = pd.read_csv(path_train)
valid = pd.read_csv(path_test) 

train = train.rename(columns={"1": "a", "2": "b" , "2.1": "c" , "15": "d" , "37": "e" , "1.1": "f" , "3": "g" , "3.1": "h" , "0": "i" , "1.2": "j" , "0.1": "k"})
valid = valid.rename(columns={"1": "a", "5": "b" , "0": "c" , "10": "d" , "34": "e" , "6": "f" , "5.1": "g" , "3": "h" , "0.1": "i" , "1.1": "j" , "0.2": "k"})

train['b'] = train['b'].div(6)
train['c'] = train['c'].div(23)
train['d'] = train['d'].div(59)
train['e'] = train['e'].div(59)

valid['b'] = valid['b'].div(6)
valid['c'] = valid['c'].div(23)
valid['d'] = valid['d'].div(59)
valid['e'] = valid['e'].div(59)

train_mod = train.to_numpy()
valid_mod = valid.to_numpy()

# X_train = train.drop(columns = ['g' , 'j'  , 'i' , 'k' , 'd' , 'e'])
# x_valid = valid.drop(columns = ['g' , 'j' , 'i' , 'k' , 'd' , 'e' ])

Y_train = train['g']
y_valid = valid['g']

X_train = train.drop(columns = ['g'])
x_valid = valid.drop(columns = ['g'])

Y_train = train['g']
y_valid = valid['g']

X_train_mod = X_train.to_numpy()
X_train_mod = X_train_mod.reshape(14742 , 1, 10)

x_valid_mod = x_valid.to_numpy()
x_valid_mod = x_valid_mod.reshape(3489 , 1, 10)

Y_train_mod = Y_train.to_numpy()

y_valid_mod = y_valid.to_numpy()


model = Sequential()

model.add(LSTM(128 , input_shape=(X_train_mod.shape[1:]), activation = 'relu' , return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128 , activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(32 , activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(8 , activation = 'relu'))
model.add(Dropout(0.2))

opt = tf.keras.optimizers.Adam(lr=1e-3 , decay = 1e-5)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

model.fit(X_train_mod , Y_train_mod , epochs= 3 , validation_data=(x_valid_mod , y_valid_mod))

# rf=RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
# rf.fit(X_train, Y_train)

# # y_train_preds = rf.predict_proba(X_train)[:,1]
# # y_valid_preds = rf.predict_proba(x_valid)[:,1]

# y_pred = rf.predict(x_valid)

# print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))