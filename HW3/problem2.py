import pandas as pd
import numpy as np

df = pd.read_csv('hw3/data/emails.csv')

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

for i in range(5):
    start = i * 1000
    end = (i + 1) * 1000
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    predictions = []
    count = 0
    for x in X_test:
        count += 1
        if count % 100 == 0:
            print(count)
        distances = [np.sqrt(np.sum((x - train_x) ** 2)) for train_x in X_train]
        nearest_neighbor_index = np.argmin(distances)
        predictions.append(y_train[nearest_neighbor_index])

    predictions = np.array(predictions)

    print(f"Fold {i}")
    true_pos = sum((predictions == 1) & (y_test == 1))
    false_pos = sum((predictions == 1) & (y_test == 0)) 
    true_neg = sum((predictions == 0) & (y_test == 0))
    false_neg = sum((predictions == 0) & (y_test == 1))
    accuracy = (true_pos + true_neg) / len(y_test)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    print(f'{true_pos} {true_neg} {false_pos} {false_neg}')
    print(f'accuracy {accuracy} precision {precision} recall {recall}')

