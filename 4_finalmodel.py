import warnings
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pickle
from sklearn.externals import joblib

warnings.filterwarnings('ignore')

df = pd.read_csv('csvs/train_after_eda.csv')

dummy = pd.get_dummies(df['Sex'])
df = pd.concat([df, dummy], axis = 1)
dummy = pd.get_dummies(df['Embarked'])
df = pd.concat([df, dummy], axis = 1)
dummy = pd.get_dummies(df['Title'])
df = pd.concat([df, dummy], axis = 1)
df = df.drop(['Sex', 'Embarked', 'Title', 'CabinAlpha'], axis = 1)

x_train = df.drop('Survived', axis = 1)
y_train = df['Survived']

logregmodel = LogisticRegression()
randomforestmodel = RandomForestClassifier()
gnbmodel = GaussianNB()
svcmodel = SVC() 
knnmodel = KNeighborsClassifier()
decisiontreemodel = DecisionTreeClassifier()

logregmodel.fit(x_train, y_train)
randomforestmodel.fit(x_train, y_train)
gnbmodel.fit(x_train, y_train)
svcmodel.fit(x_train, y_train)
knnmodel.fit(x_train, y_train)
decisiontreemodel.fit(x_train, y_train)

joblib.dump(logregmodel, 'trainedmodels/logregmodel.pkl')
joblib.dump(randomforestmodel, 'trainedmodels/randomforestmodel.pkl')
joblib.dump(gnbmodel, 'trainedmodels/gnbmodel.pkl')
joblib.dump(svcmodel, 'trainedmodels/svcmodel.pkl')
joblib.dump(knnmodel, 'trainedmodels/knnmodel.pkl')
joblib.dump(decisiontreemodel, 'trainedmodels/decisiontreemodel.pkl')

y_predict_lgm = logregmodel.predict(x_train)
y_predict_rfm = randomforestmodel.predict(x_train)
y_predict_gnb = gnbmodel.predict(x_train)
y_predict_svc = svcmodel.predict(x_train)
y_predict_knn = knnmodel.predict(x_train)
y_predict_dt = decisiontreemodel.predict(x_train)

"""
metrics.accuracy_score(y_predict_lgm, y_test)
metrics.accuracy_score(y_test, y_predict_rfm)
metrics.accuracy_score(y_test, y_predict_gnb)
metrics.accuracy_score(y_test, y_predict_svc)
metrics.accuracy_score(y_test, y_predict_knn)
metrics.accuracy_score(y_test, y_predict_dt)
"""

y_pred = np.stack((y_predict_dt, y_predict_gnb, y_predict_knn, y_predict_lgm, y_predict_rfm, y_predict_svc), axis = 1)
df1 = pd.DataFrame(y_pred, columns = ['dt', 'gnb', 'knn', 'lgm', 'rfm', 'svc'])

#Ensembling using logistic regression
model = LogisticRegression()
model.fit(df1, y_train)

joblib.dump(model, 'trainedmodels/ensemblemodel.pkl')



