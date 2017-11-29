import numpy as np
from sklearn.metrics import accuracy_score

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# DAta and labels [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#classifiers
clf_SVC = svm.SVC(kernel='rbf')
clf_RF  = RandomForestClassifier(n_estimators=10 , max_depth = 2)
clf_Knn = KNeighborsClassifier(n_neighbors = 3 )
clf_tree = tree.DecisionTreeClassifier()

#fitting the data set
clf_SVC = clf_SVC.fit(X, Y)
clf_RF = clf_RF.fit(X, Y)
clf_Knn = clf_Knn.fit(X, Y)
clf_tree = clf_tree.fit(X, Y)

prediction = clf_tree.predict([[190, 70, 43]])
print(prediction)

#predicting
pred_tree = clf_tree.predict(X)
acc_tree = accuracy_score(Y,pred_tree) *100
print('Decision Tree: {} '.format(acc_tree))

pred_SVC = clf_SVC.predict(X)
acc_SVC = accuracy_score(Y,pred_SVC) *100
print('Support Vector Machine: {} '.format(acc_SVC))

pred_RF = clf_RF.predict(X)
acc_RF = accuracy_score(Y,pred_RF) *100
print('Random Forest : {} '.format(acc_RF))

pred_knn = clf_Knn.predict(X)
acc_knn = accuracy_score(Y,pred_knn) *100
print('KNN : {} '.format(acc_knn))
