import pandas
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense

df_main=pandas.read_csv("combined_new_forhome.csv")

#df_main_final=df_main[df_main["FTR_int"] != 0].reset_index(drop=True)

X = df_main.iloc[:, 7:11].values
y = df_main.iloc[:, 11].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#labelencoder_y = LabelEncoder()
#y_train = labelencoder_y.fit_transform(y_train)
#y_test = labelencoder_y.transform(y_test)

selection_algorithm_list={"SVC": SVC, "KNeighborsClassifier" : KNeighborsClassifier, "LogisticRegression": LogisticRegression,
                          "GaussianNB": GaussianNB, "DecisionTreeClassifier" : DecisionTreeClassifier,
                          "RandomForestClassifier": RandomForestClassifier, "XGBClassifier": XGBClassifier, "ANN": Sequential}

df_test_results=pandas.DataFrame()

hellas=[]
y_predder=[]
y_psudo=[]
for key, value in selection_algorithm_list.items():
    if key == "SVC":
        classifier= value(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
    elif key == "KNeighborsClassifier":
        classifier = value(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
    elif key == "DecisionTreeClassifier":
        classifier= value(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
    elif key == "RandomForestClassifier":
        classifier= value(n_estimators = 100, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
    elif key == "LogisticRegression":
        classifier= value(random_state=0)
        classifier.fit(X_train, y_train)
    elif key == "GaussianNB":
        classifier= value()
        classifier.fit(X_train, y_train)
    elif key == "XGBClassifier":
        classifier = value()
        classifier.fit(X_train, y_train)
    elif key == "ANN":
        classifier = value()
        classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 4))
        classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
    
    y_pred = classifier.predict(X_test)
    y_psudo_pred = classifier.predict(X_train)
    y_psudo.append(y_psudo_pred)
    y_predder.append(y_pred)
    try:
        cm = confusion_matrix(y_test, y_pred)
    except:
        y_pred=(y_pred>0.5)
        cm = confusion_matrix(y_test, y_pred)
    
    #accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
    
    try:
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
        accuracies_mean=accuracies.mean()
    except:
        accuracies_mean=(cm[0][0]+cm[1][1])/len(X_test)
        #accuracies_mean=0
        #a=0
        #accuracies=np.asarray(a)
    
    #df_test_results[key]= [confusion_matrix(y_test,y_pred)]
    df_test_results[key]= [cm,accuracies_mean]
    #hellas.append([confusion_matrix(y_test,y_pred),accuracies.mean()])
