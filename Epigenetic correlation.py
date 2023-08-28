import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
from matplotlib.font_manager import FontProperties  
from matplotlib import pyplot
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.decomposition import IncrementalPCA
import csv
from sklearn.preprocessing import StandardScaler  

def clean(df):  
    c_names = df.columns.values.tolist()   
    for c_name in c_names:  
        df[c_name]=pd.to_numeric(df[c_name],'coerce') 
    df=df.dropna(axis = 0)  
    df=df.reset_index(drop=True)   
    return df

jbname='leukemia' 
path_dfs = r'leukemia-train-examples.csv' 
dfs=pd.read_csv(path_dfs)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

# data0 = pd.read_csv(r'......csv',encoding='gbk')
dfs1 = dfs[dfs['label']==1].iloc[:,:]
dfs1['label']=('experimental group ')
dfs0 = dfs[dfs['label']==0].iloc[:,:]
dfs0['label']=('control group')
 
whole=pd.concat([dfs1, dfs0], axis=0, ignore_index=True)
whole=whole.iloc[:50,1:]

whole.columns=['sex','age','RBC', 'MCV', 'PDW', 'WBC', 'NEUT%', 'LYMPH%', 'EO%', 'BASO%', 'NEUT#', 'LYMPH#', 'BASO#', 'HGB', 'HCT', 'MCH', 'MCHC', 'R-CV', 'PLT', 'MPV', 'PCT', 'MONO#', 'MONO%', 'EO#','标签']

kind_dict = {
    "control group":"control group",
    "experimental group ":"experimental group "
}
whole["label"] = whole["label"].map(kind_dict)
sns.pairplot(whole,hue="label",plot_kws=dict(s=3))
plt.savefig(r'scatter diagram '+'.jpg', dpi=300)
plt.show()