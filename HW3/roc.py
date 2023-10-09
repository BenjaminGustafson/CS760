import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

confidences = np.array([0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1])
labels = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 0])

fpr, tpr, thresholds = roc_curve(labels, confidences)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
