from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import pandas as pd

#read data
data = pd.read_csv('sift_230.csv')

#data preprocessing
X = data.drop(['class'], axis = 1)
X = X.drop(['image_name'], axis = 1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42, stratify = y)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#model selection ({'C': 64, 'gamma': 0.00390625})
params = {
    'C' : [2**x for x in range(-12, 13)],
    'gamma' : [2**x for x in range(-12, 13)]
}

clf = GridSearchCV(svm.SVC(), params, cv = 5, n_jobs = 4)
clf.fit(X_train, y_train)

print(clf.best_params_)

#train algorithm
model = svm.SVC(C = clf.best_params_['C'], gamma = clf.best_params_['gamma'])
model.fit(X_train, y_train)

#model evaluation on the test data (score = 0.8461538461538461)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#model evaluation using cross validation (score = 0.850645404254214)
model = svm.SVC(C = clf.best_params_['C'], gamma = clf.best_params_['gamma'])

scaler.fit(X)
X = scaler.transform(X)

print(cross_val_score(model, X, y, cv = 5).mean())