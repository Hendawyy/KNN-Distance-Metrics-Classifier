import pandas as pd

Traindataset = pd.read_csv('FacesFeaturesTrain.csv')
Testdataset = pd.read_csv('FacesFeaturesTest.csv')

x_train = Traindataset.iloc[:, 0:64].values
y_train = Traindataset.iloc[:, 64].values

x_test = Testdataset.iloc[:, 0:64].values
y_test = Testdataset.iloc[:, 64].values

#Using Feature Scaling to Scale Values in X To Numerical Values Between -1.5 & 1.5.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

#Import PCA
from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.85)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
variance = pca.explained_variance_ratio_
explained_variance = pca.explained_variance_ratio_

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

#Optimal Value Of K
acc = []

for i in range(1,10):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(x_train,y_train)
    yhat = neigh.predict(x_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,10),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')

K=3
#KNN with K = Optimum Using The Distance Equation Minkowski.
ClassifierM = KNeighborsClassifier(n_neighbors=K, metric ='minkowski', p=2)
ClassifierM.fit(x_train, y_train)
y_pred = ClassifierM.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Minkowski.
k5cfM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix : ')
print(k5cfM)

#Classification Report For K = Optimum Using The Distance Equation Minkowski.
k5crM = classification_report(y_test, y_pred)
print('Classification Report : ')
print(k5crM)

#Accuracy Score For K = Optimum Using The Distance Equation Minkowski.
k5asM = accuracy_score(y_test, y_pred)
print('Accuracy :',k5asM*100,'%')



#KNN with K = Optimum Using The Distance Equation Euclidean.
ClassifierE = KNeighborsClassifier(n_neighbors=5, metric ='euclidean')
ClassifierE.fit(x_train, y_train)
y_predE5 = ClassifierE.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Euclidean.
k5cfE = confusion_matrix(y_test, y_predE5)
print('Confusion Matrix : ')
print(k5cfE)

#Classification Report For K = Optimum Using The Distance Equation Euclidean.
k5crE = classification_report(y_test, y_predE5)
print('Classification Report : ')
print(k5crE)

#Accuracy Score For K = Optimum Using The Distance Equation Euclidean.
k5asE = accuracy_score(y_test, y_predE5)
print('Accuracy :',k5asE*100,'%')



#KNN with K = Optimum Using The Distance Equation Manhattan.
ClassifierN = KNeighborsClassifier(n_neighbors=K, metric ='manhattan')
ClassifierN.fit(x_train, y_train)
y_predN5 = ClassifierN.predict(x_test)


#Confusion Matrix For K = Optimum Using The Distance Equation Manhattan.
k5cfN = confusion_matrix(y_test, y_predN5)
print('Confusion Matrix : ')
print(k5cfN)

#Classification Report For K = Optimum Using The Distance Equation Manhattan.
k5crN = classification_report(y_test, y_predN5)
print('Classification Report : ')
print(k5crN)

#Accuracy Score For K = Optimum Using The Distance Equation Manhattan.
k5asN = accuracy_score(y_test, y_predN5)
print('Accuracy :',k5asN*100,'%')




#Intialize Data to be Put in The Dafarame. 
K5C = {'Confusion Matrix' : pd.Series([k5cfM, k5cfE, k5cfN], index =['Minkowski', 'Euclidean', 'Manhattan']),
      'Accuracy Score' : pd.Series([k5asM, k5asE, k5asN], index =['Minkowski', 'Euclidean', 'Manhattan'])} 

#Create an Comparison Dataframe That Holds a comparison between The Accuracy Scores and the confusion matrices of the diffrent Distance Functions. 
K5 = pd.DataFrame(K5C) 

print('K = Optimum\n',K5)
