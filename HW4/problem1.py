import numpy as np

def multinomial(theta):
    i = 0
    threshold = 0
    r = np.random.rand()
    while r > threshold:
        threshold += theta[i]
        i += 1
    return i-1

theta = np.array([0.4,0.2,0.2,0.1,0.1])
histo = np.zeros(len(theta))
strat1 = 0
strat2 = 0
N = 1000000
for i in range(N):
    x = multinomial(theta)
    histo[x] += 1
    if x != 0:
        strat1 += 1
    if multinomial(theta) != x:
        strat2 += 1

probs = histo / N
print(f"Observed distribution: {probs}")
print(f"Strategy 1 loss: {strat1/N}. Should be {1 - max(theta)}")
print(f"Strategy 2 loss: {strat2/N}. Should be {1 - np.square(theta).sum()}")