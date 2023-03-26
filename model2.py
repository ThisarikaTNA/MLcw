from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from main import X_train, y_train, X_test, y_test

# Instantiate the Decision Trees model
dt = DecisionTreeClassifier(random_state=42)

# Train the Decision Trees model
dt.fit(X_train, y_train)

# Predict the labels for the testing set using the trained Decision Trees model
y_pred_dt = dt.predict(X_test)

# Evaluate the performance of the Decision Trees model
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
print('Decision Trees Model Performance')
print('Accuracy:', acc_dt)
print('Precision:', prec_dt)
print('Recall:', rec_dt)
print('F1-score:', f1_dt)