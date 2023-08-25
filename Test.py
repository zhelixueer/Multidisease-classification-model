#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
import joblib
jbname='leukemia'
path_test = r'leukemia-train-examples.csv' 
data_test=pd.read_csv(path_test)
data_test=data_test.iloc[:500,1:]
y_test2 =data_test['label']

data2= data_test.drop(['label'],axis=1,inplace=False).values[:,:]
data_test.drop(['label'],axis=1,inplace=False)

from sklearn.metrics import classification_report  
from sklearn.metrics import confusion_matrix

ss_model=joblib.load(jbname+'Scaler.pkl')  # Standardized model
data2=ss_model.transform(data2)
from tensorflow.keras.models import load_model
model = load_model(jbname+'.h5')
predict=model.predict_classes(data2).astype('int')
alfa = 0.5
predict_ce = predict > alfa   # alfa 
print('Regulate the threshold', alfa)
target_names = ['class 0', 'class 1']
print(classification_report(y_test2, predict, target_names=target_names))
con = confusion_matrix(y_test2, predict)
print(con)
tn = con[0][0]
fn = con[1][0]
tp = con[1][1]
fp = con[0][1]
zhenyang=(tp / (fn + tp))
zhenyin=(tn / (tn + fp))
jiayang=(fp / (fp + tn))
jiayin=(fn / (fn + tp))
print('TP',zhenyang)
print('TN',zhenyin)
print('FP',jiayang)
print('FN',jiayin)




