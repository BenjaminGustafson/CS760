import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


df = pd.read_csv('hw3/data/emails.csv')

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

X = np.hstack((np.ones((X.shape[0], 1)), X))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(features, labels, learning_rate=0.001, num_epochs=1000):
    num_samples, num_features = features.shape
    weights = np.zeros(num_features)
    
    for epoch in range(num_epochs):
        predictions = sigmoid(np.dot(features, weights))
        gradient = np.dot(features.T, (predictions - labels)) / num_samples
        weights -= learning_rate * gradient
    
    return weights


start = 0
end = 1000
X_test = X[start:end]
y_test = y[start:end]
X_train = np.concatenate((X[:start], X[end:]), axis=0)
y_train = np.concatenate((y[:start], y[end:]), axis=0)

weights = logistic_regression(X_train, y_train)

logistic_confidences = sigmoid(np.dot(X_test, weights))

knn_preds = []
knn_confidences = []
count = 0
for x in X_test:
    count += 1
    if count % 100 == 0:
        print(count)
    distances = [np.sqrt(np.sum((x - train_x) ** 2)) for train_x in X_train]
    nearest_neighbor_indices = np.argsort(distances)[:5]
    # unique_values, counts = np.unique(y_train[nearest_neighbor_indices], return_counts=True)
    # prediction = unique_values[np.argmax(counts)]
    # knn_preds.append(prediction)
    knn_confidences.append(np.mean(y_train[nearest_neighbor_indices]))


log_fpr, log_tpr, log_thresholds = roc_curve(y_test, logistic_confidences)

knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_confidences)

plt.figure(figsize=(6, 6))
plt.plot(log_fpr, log_tpr, color='red', lw=2)
plt.plot(knn_fpr, knn_tpr, color='blue', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
