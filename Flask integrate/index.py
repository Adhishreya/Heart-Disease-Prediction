import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest,chi2,f_classif
from sklearn.model_selection import  train_test_split,KFold,GridSearchCV,StratifiedShuffleSplit,cross_val_score,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,plot_confusion_matrix
import pickle
import os
data=pd.read_csv('./heart.csv')
data.head()
Y=data['target']
X=data.drop('target',axis=1)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
scaler=StandardScaler().fit(x_train)
X_train=scaler.transform(x_train)
X_test=scaler.transform(x_test)

log=LogisticRegression(max_iter=500,solver='newton-cg').fit(X_train,y_train)
y_pred=log.predict(X_test)
log_conf=plot_confusion_matrix(log,X_test,y_test,cmap=plt.cm.Blues,normalize='true')
print("Model Accuracy",format(accuracy_score(y_test,y_pred)*100,".3f"),"%")

pickle.dump(log, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[4, 300, 500]]))