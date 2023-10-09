import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

short = False

df = pd.read_csv('hw3/data/emails.csv')

X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
if short:
    X=df.iloc[:100, 1:100].values
    y = df.iloc[:100, -1].values

ks = [1,3,5,7,10]

accuracies = [[] for k in ks]
num_folds = 5
for i in range(num_folds):
    start = i * len(y) // num_folds 
    end = (i + 1) * len(y) // num_folds 
    X_test = X[start:end]
    y_test = y[start:end]
    X_train = np.concatenate((X[:start], X[end:]), axis=0)
    y_train = np.concatenate((y[:start], y[end:]), axis=0)

    predictions = [[] for k in ks]
    count = 0
    for x in X_test:
        count += 1
        if count % 100 == 0:
            print(count)
        distances = [np.sqrt(np.sum((x - train_x) ** 2)) for train_x in X_train]
        for k in ks:
            nearest_neighbor_indices = np.argsort(distances)[:k]
            unique_values, counts = np.unique(y_train[nearest_neighbor_indices], return_counts=True)
            prediction = unique_values[np.argmax(counts)]
            predictions[ks.index(k)].append(prediction)

    predictions = np.array(predictions)
    print("Predictions")
    print(predictions)
    print("Labels")
    print(y_test)

    for k in ks:
        print(f"________ K = {k} _____________")
        print(f"Fold {i}")
        preds_k = predictions[ks.index(k)]
        true_pos = sum((preds_k == 1) & (y_test == 1))
        false_pos = sum((preds_k == 1) & (y_test == 0)) 
        true_neg = sum((preds_k == 0) & (y_test == 0))
        false_neg = sum((preds_k == 0) & (y_test == 1))
        accuracy = (true_pos + true_neg) / len(y_test)
        print(f'{true_pos} {true_neg} {false_pos} {false_neg}')
        print(f'accuracy {accuracy}')
        accuracies[ks.index(k)].append(accuracy)

average_accuracies = []
for (i,k) in enumerate(ks):
    avg_accuracies_k = np.mean(np.array(accuracies[i]))
    print(f"k {k} average accuracy {avg_accuracies_k:.3f}")
    average_accuracies.append(avg_accuracies_k)


plt.plot(ks, average_accuracies)
plt.xlabel('k')
plt.ylabel('Average accuracy')
plt.show()
