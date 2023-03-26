from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from main import X_train, y_train, X_test, y_test

# Instantiate the KNN model
knn = KNeighborsClassifier()

# Train the KNN model
knn.fit(X_train, y_train)

# Predict the labels for the testing set using the trained KNN model
y_pred_knn = knn.predict(X_test)

# Evaluate the performance of the KNN model
acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
print('KNN Model Performance')
print('Accuracy:', acc_knn)
print('Precision:', prec_knn)
print('Recall:', rec_knn)
print('F1-score:', f1_knn)