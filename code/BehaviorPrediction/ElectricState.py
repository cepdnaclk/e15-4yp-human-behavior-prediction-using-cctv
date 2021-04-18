import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier,Pool
from sklearn import metrics



train = pd.read_csv("train.csv", 
                  sep=',', 
                  names=["ID", "Day", "Time(Hours)", "Time(Minutes)","Time(Seconds)","Location", "Next Destination", "Action","Bathroom Light","Kitchen Light","TV"])
valid = pd.read_csv("test.csv", 
                  sep=',', 
                  names=["ID", "Day", "Time(Hours)", "Time(Minutes)","Time(Seconds)","Location", "Next Destination" , "Action","Bathroom Light","Kitchen Light","TV"])


features=["ID", "Day", "Time(Hours)", "Time(Minutes)","Time(Seconds)","Location","Next Destination" , "Action","Bathroom Light","Kitchen Light","TV"]
Xfeatures=["ID", "Day", "Time(Hours)", "Time(Minutes)", "Location" , "Action"]
x_train=train[Xfeatures]
x_valid=valid[Xfeatures]
y_valid=valid["TV"]
y_train=train["TV"]


y_valid=valid["Bathroom Light"]
y_train=train["Bathroom Light"]


rf=CatBoostClassifier(iterations=20,
                      learning_rate=0.003,
                      eval_metric='AUC',
                      nan_mode='Min',
                      thread_count=8,
                      task_type='CPU',
                      verbose=True)

rf.fit(x_train, y_train)


y_pred = rf.predict(x_valid)

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred))

