from utils.preprocessing import load_data, scale_data
from models.dnn import build_dnn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load data
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scale_data(X_train, X_test)

# Train model
model = build_dnn(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, verbose=0)

# Predictions
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print("\n RESULTS")
print("Accuracy:", round(accuracy, 3))
print("ROC-AUC:", round(roc_auc, 3))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))