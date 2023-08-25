#!/usr/bin/env python
# coding: utf-8

import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler  # Standardized processing
def clean(df):  # Rines containing non-numerical values were removed
    print('Raw sample size before containing nonvalues were removed',len(df))  
    c_names = df.columns.values.tolist()   
    for c_name in c_names:   # List-by-list investigation
        df[c_name]=pd.to_numeric(df[c_name],'coerce')  # Empty value instead
    df=df.dropna(axis = 0)  # Delete the null value by line
    df=df.reset_index(drop=True)   # Recompile the index number
    print('The remaining valid sample size remains after removing non-numbers',len(df))
    return df

jbname='leukemia' 
path_train = r'leukemia-train-examples.csv' 
data_train=pd.read_csv(path_train)
data_train=data_train.iloc[:,1:]

data_x= data_train.drop(['label'],axis=1,inplace=False).values[:,:]
ss_x=StandardScaler()
ss_x.fit_transform(data_x)

import joblib   # Model preservation
joblib.dump(ss_x, jbname+'Scaler.pkl')


import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
import pandas as pd
import numpy as np
import os
import datetime
from tensorflow.keras import regularizers
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing, metrics
from sklearn.metrics import classification_report  

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib as plt
from sklearn.metrics import accuracy_score

from keras.callbacks import TensorBoard

d=data_train
#  Separate the 0,1 data in the raw data
zerodata = d[d['label']==0]
onedata = d[d['label']==1]
print('Original 0 class',len(zerodata))
print('原始1类',len(onedata))
pit=len(onedata)/len(zerodata)
print('Original class 1/0类',pit)
#TODO:Class 0 cuts were divided into training and test sets
xtest0 = zerodata.sample(frac=0.2,replace=False,random_state=None)######## No back   frac    n
xtrain0 =zerodata[~zerodata.index.isin(xtest0.index)]

#TODO:Class 1 was split into training and test sets
xtest1 = onedata.sample(frac=0.2,replace=False,random_state=None)########### No back
xtrain1 =onedata[~onedata.index.isin(xtest1.index)]

tmp = len(xtrain0)/len(xtrain1)
tmp = 1  ############
#TODO:Moled to generate final training and test data
testdata = pd.concat([xtest0,xtest1],axis=0,ignore_index=True)
traindata = pd.concat([xtrain0,xtrain1],axis=0,ignore_index=True)
traindata = pd.concat([xtrain1.sample(frac=1*tmp,replace=True,random_state=None),xtrain0.sample(frac=1,replace=True,random_state=None)],axis=0,ignore_index=True)#########
print('Training set 0 classes',len(traindata[traindata['label']==0]))
print('Training set category 1',len(traindata[traindata['label']==1]))
p=len(traindata[traindata['label']==0])/len(traindata[traindata['label']==1])
print('The 0 / 1 ratio of the training set:',1/p)
X_train = traindata.drop(['label'],axis=1,inplace=False)
y_train = traindata['label']
X_test = testdata.drop(['label'],axis=1,inplace=False)
y_test = testdata['label']
########################################################
X_train=ss_x.transform(X_train)
X_test=ss_x.transform(X_test)

# neural network：
model = keras.Sequential([
        layers.Dense(1024, activation='relu', input_shape=[data_train.shape[1]-1],name='input_dense'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid',name='output_dense'),
    ])

    # Define the loss function and the optimization algorithm for the model
model.compile( 
        optimizer='Adam',   # SGD   Adam
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )   

def fit(model, X_train, y_train):
    H = model.fit(
        class_weight={0:p, 1:1},  
        x=X_train,
        y=y_train,
        batch_size=1024,
        epochs=50,
#         callbacks=[tbCallBack]
    )
    return model,H

# tbCallBack = TensorBoard()
model, H = fit(model, X_train, y_train)

print('Number of training rounds',len(H.history['loss']))

train_score = model.evaluate(X_train, y_train)
print("Train loss:", train_score[0])
print("Train accuracy:", train_score[1])
test_scores = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
model.save(jbname+'.h5')

model.save(jbname+'.h5')


# predict = model.predict_classes(X_test).astype('int')
alfa = 0.5
predict = model.predict(X_test)>alfa
print('Regulate the threshold', alfa)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, predict, target_names=target_names))
con = confusion_matrix(y_test, predict)
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
print('YN',zhenyin)
print('FP',jiayang)
print('FN',jiayin)
# ————————————————————————————————————————————————

# ROC&AUC
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
y_one_hot = label_binarize(y_test, np.arange(2))
fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),model.predict_proba(X_test).ravel())
auc = metrics.auc(fpr, tpr)
print ('auc：', auc)
mpl.rcParams['font.sans-serif'] = u'Microsoft YaHei'
mpl.rcParams['axes.unicode_minus'] = False
font=15
plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
plt.rc('figure', figsize=(5, 5))
plt.plot((0, 1), (0, 1), c = '#808080', lw = 2, ls = '--', alpha = 1.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=font)
plt.ylabel('True Positive Rate', fontsize=font)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=font)
plt.title(u'ROC&AUC', fontsize=font)
plt.show()

import shap
explainer = shap.KernelExplainer(model.predict,X_test)
shap_values = explainer.shap_values(X_test,nsamples=20)
fig = plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.tight_layout()
plt.show()




