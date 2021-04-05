import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path_train = 'train.csv'
path_test = 'test.csv'

train = pd.read_csv(path_train)
valid = pd.read_csv(path_test) 

train = train.rename(columns={"1": "a", "2": "b" , "2.1": "c" , "15": "d" , "37": "e" , "1.1": "f" , "3": "g" , "3.1": "h" , "0": "i" , "1.2": "j" , "0.1": "k"})
valid = valid.rename(columns={"1": "a", "5": "b" , "0": "c" , "10": "d" , "34": "e" , "6": "f" , "5.1": "g" , "3": "h" , "0.1": "i" , "1.1": "j" , "0.2": "k"})


X_train = train.drop(columns = ['g' , 'j'  , 'i' , 'k' , 'd' , 'e'])
x_valid = valid.drop(columns = ['g' , 'j' , 'i' , 'k' , 'd' , 'e' ])

Y_train = train['g']
y_valid = valid['g']

rf=RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
rf.fit(X_train, Y_train)

# y_train_preds = rf.predict_proba(X_train)[:,1]
# y_valid_preds = rf.predict_proba(x_valid)[:,1]

y_pred = rf.predict(x_valid)

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))