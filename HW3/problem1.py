import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def read_data(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        for line in file:
            x1, x2, label = map(float, line.strip().split())
            X.append((x1, x2))
            y.append(label)
    return (np.array(X),np.array(y))

(X,y) = read_data('HW3/data/D2z.txt')


xx, yy = np.meshgrid(np.arange(-2, 2.1, 0.1), np.arange(-2, 2.1, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

predictions = []
for point in grid_points:
    distances = [np.sqrt(np.sum((point - x) ** 2)) for x in X]
    nearest_neighbor = np.argmin(distances)
    predictions.append(y[nearest_neighbor])

predictions = np.array(predictions)

plt.scatter(grid_points[:, 0], grid_points[:, 1], c=np.where(predictions == 1, 'red', 'blue'), s=10, edgecolors='none')

plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o',facecolors='none' ,color='black', label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='black', label='1')


plt.show()
