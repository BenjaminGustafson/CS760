import pandas as pd
import numpy as np

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

for i in range(5):
    start = i * 1000
    end = (i + 1) * 1000
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    weights = logistic_regression(X_train, y_train)
    
    predictions = (sigmoid(np.dot(X_test, weights)) >= 0.5).astype(int)

    print(f"Fold {i}")
    true_pos = sum((predictions == 1) & (y_test == 1))
    false_pos = sum((predictions == 1) & (y_test == 0)) 
    true_neg = sum((predictions == 0) & (y_test == 0))
    false_neg = sum((predictions == 0) & (y_test == 1))
    accuracy = (true_pos + true_neg) / len(y_test)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    print(f'{true_pos} {true_neg} {false_pos} {false_neg}')
    print(f'accuracy {accuracy:.3f} precision {precision:.3f} recall {recall:.3f}')

