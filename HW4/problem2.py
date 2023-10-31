import numpy as np

def multinomial(theta):
    i = 0
    threshold = 0
    r = np.random.rand()
    while r > threshold:
        threshold += theta[i]
        i += 1
    return i-1

theta = np.array([0.6,0.3,0.1])
c = np.array([[0,1,1],[1,0,1],[10,1,0]])
histo = np.zeros(len(theta))
loss = 0
N = 1000000
min = 10000
argmin = 0
for j in range(len(theta)):
    res = 0 
    for i in range(len(theta)):
        res += theta[i] * c[i][j]
    if res < min:
        min = res
        argmin = j
    print(f"Predicting {j} gives expected loss {res}.")

for i in range(N):
    x = multinomial(theta)
    histo[x] += 1
    x_pred = argmin
    loss += c[x][x_pred]
loss = loss/N    

probs = histo / N
print(f"Observed distribution: {probs}")
print(f"Loss: {loss}. Should be {min}.")
