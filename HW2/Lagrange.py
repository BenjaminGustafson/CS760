import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt


"""
Fix some interval [a, b] and sample n = 100 points x from this interval uniformly. Use these to build a training
set consisting of n pairs (x, y) by setting function y = sin(x).
"""

a, b = 0, 1  
n = 100  

x_train = np.linspace(a, b, n)
y_train = np.sin(x_train)


"""
Build a model f by using Lagrange interpolation, check more details in https://en.wikipedia.org/wiki/Lagrange
polynomial and https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html.
"""

lagrange_model = lagrange(x_train, y_train)

"""
Generate a test set using the same distribution as your test set. Compute and report the resulting model’s train and
test error. What do you observe?
"""

x_test = np.linspace(a, b, n)
y_test = np.sin(x_test)

train_error = np.mean((lagrange_model(x_train) - y_train)**2)
test_error = np.mean((lagrange_model(x_test) - y_test)**2)

print("Training Error:", train_error)
print("Test Error:", test_error)


"""
 Repeat the experiment with zero-mean Gaussian noise ε added to x. Vary the
standard deviation for ε and report your findings.
"""

std_dev_list = [0.001, 0.01, 0.1, 1.0, 10.0] 

for std_dev in std_dev_list:
    noisy_x_test = x_test + np.random.normal(0, std_dev, n)
    y_test = np.sin(noisy_x_test)
        
    model = lagrange(noisy_x_test, y_test)

    print(f"\n Standard deviation: {std_dev}")
    print("Train error:", np.mean((model(noisy_x_test) - y_test)**2))
    print("Test error:",  np.mean((model(x_test) - y_test)**2))
    