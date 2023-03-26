from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from main import X_train, y_train, X_test, y_test

#enhance

# Instantiate the KNN model
knn = KNeighborsClassifier()

# Define the parameter grid to search over for the KNN model
knn_params = {'n_neighbors': [3, 5, 7, 9]}

# Perform a grid search to find the best hyperparameters for the KNN model
knn_grid = GridSearchCV(knn, knn_params, cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_best = knn_grid.best_estimator_

# Train the KNN model with the best hyperparameters
knn_best.fit(X_train, y_train)

# Predict the labels for the testing set using the trained KNN model
y_pred_knn = knn_best.predict(X_test)

# Evaluate the performance of the KNN model
acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

print('KNN Model Performance')
print('Accuracy:', acc_knn)
print('Precision:', prec_knn)
print('Recall:', rec_knn)
print('F1-score:', f1_knn)
print('Confusion Matrix:')
print(conf_matrix_knn)

# Instantiate the Decision Trees model
dt = DecisionTreeClassifier(random_state=42)

# Define the parameter grid to search over for the Decision Trees model
dt_params = {'max_depth': [3, 5, 7, 9], 'min_samples_split': [2, 5, 10]}

# Perform a grid search to find the best hyperparameters for the Decision Trees model
dt_grid = GridSearchCV(dt, dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train, y_train)
dt_best = dt_grid.best_estimator_

# Train the Decision Trees model with the best hyperparameters
dt_best.fit(X_train, y_train)

# Predict the labels for the testing set using the trained Decision Trees model
y_pred_dt = dt_best.predict(X_test)

# Evaluate the performance of the Decision Trees model
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print('Decision Trees Model Performance')
print('Accuracy:', acc_dt)
print('Precision:', prec_dt)
print('Recall:', rec_dt)
print('F1-score:', f1_dt)
print('Confusion Matrix:')
print(conf_matrix_dt)