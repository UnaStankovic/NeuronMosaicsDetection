from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

import pandas as pd

#read data, and balance classes
data = pd.read_csv('data.csv')
print(data[data['class'] == False].shape[0])
print(data[data['class'] == True].shape[0])

data_new = data[data['class'] == True]
data_new = data_new.append(data[data['class'] == False][0:493], ignore_index = True)
data = data_new

#data preprocessing
X = data.drop(['class'], axis = 1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42, stratify = y)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#model selection ({'max_features': 36, 'min_samples_leaf': 1})
params = {
    'min_samples_leaf' : [x for x in range(1, 6)],
    'max_features' : [x for x in range(5, 37)]
}

clf = GridSearchCV(RandomForestClassifier(random_state = 42), params, cv = 5, n_jobs = 4)
clf.fit(X_train, y_train)

print(clf.best_params_)

#train algorithm
model = RandomForestClassifier(min_samples_leaf = clf.best_params_['min_samples_leaf'], max_features = clf.best_params_['max_features'], random_state = 42)
model.fit(X_train, y_train)

#model evaluation on the test data (score = 0.8412162162162162)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

#model evaluation using cross validation (score = 0.8112760255617399)
model = RandomForestClassifier(min_samples_leaf = clf.best_params_['min_samples_leaf'], max_features = clf.best_params_['max_features'], random_state = 42)

scaler.fit(X)
X = scaler.transform(X)

print(cross_val_score(model, X, y, cv = 5).mean())
