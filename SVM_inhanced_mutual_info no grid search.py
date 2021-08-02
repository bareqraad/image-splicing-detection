from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
#from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

No_of_Features = 508
filePath='f:/cb10.csv'
k=50

#param_grid = {'C': [1], 'gamma': ['auto'],'kernel': ['linear','rbf', 'poly', 'sigmoid']}
#param_grid = {'C': [0.003], 'gamma': ['auto'],'kernel': ['rbf']}
def select_features(X_train,y_train,X_test):
     fs = SelectKBest(score_func=mutual_info_classif, k=k)
     fs.fit(X_train, y_train)     
     X_train_fs = fs.transform(X_train)
     X_test_fs  = fs.transform(X_test)
     return X_train_fs,X_test_fs,fs 
data = pd.read_csv(filePath) 
d=data.values
X = d[:,0:No_of_Features-1]
y = d[:,No_of_Features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,shuffle = True)
X_train_fs, X_test_fs, fs =select_features(X_train,y_train,X_test)
# model = GridSearchCV(SVC(),param_grid,cv=5,refit=True,verbose=2,n_jobs=-1)
model = SVC()
model.fit(X_train_fs, y_train)
y_pred = model.predict(X_test_fs)

# print('Best GridSearchCV parameters: ',model.best_params_)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



import pickle
# save the model to disk
filename = 'f:/data.sav'
pickle.dump(model, open(filename, 'wb'))


# load the model from disk
model = pickle.load(open(filename, 'rb'))
result = model.score(X_test, y_test)
print(result)


print("%%%%%%%%%%%%%%predict new data%%%%%%%%%%%%%%%%%")
data = pd.read_csv('f:/test.csv') 
d=data.values
X = d[:,0:No_of_Features-1]
xfs = fs.transform(X)
z = model.predict(xfs)
print(z)
