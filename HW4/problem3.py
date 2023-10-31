import numpy as np


vocab = 'abcdefghijklmnopqrstuvwxyz '

X = []
y = []

for i in range(20):
    for (c,lang) in enumerate('ejs'):
        char_counts = [0] * 27
        with open(f'HW4/languageID/{lang}{i}.txt', 'r') as file:
            content = file.read()
            for char in content:
                if char in vocab:
                    char_counts[vocab.index(char)] += 1
        X.append(char_counts)
        y.append(c)

X = np.array(X)
y = np.array(y)

num_classes = 3
num_chars = 27

split = 10*num_classes
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

prior_probs = [1/3]*3 # all equal since same number of documents for each class
log_likelihoods = np.zeros((num_classes, num_chars))

for (c,lang) in enumerate('ejs'):
    char_counts = X_train[y_train == c].sum(axis=0) + 0.5
    total_chars = char_counts.sum()
    log_likelihoods[c] = np.log(char_counts) - np.log(total_chars)
    print(f"Class {lang} likelihoods {np.exp(log_likelihoods[c])}")
    
for i in range(27):
    print(f"{i} & {vocab[i]} &  {np.exp(log_likelihoods[0][i]):.5f} & {np.exp(log_likelihoods[1][i]):.5f} & {np.exp(log_likelihoods[2][i]):.5f}\\\\")

print("Part 4")
print(f"double check {y[30]} should be 0")
for i in range(27):
    print(f"{vocab[i]} & {X[30][i]} \\\\")

print("Part 5&6")
predictions = []
confusion = np.zeros((3,3))
for i in range(len(X_test)):
    doc_probs = []
    for c in range(num_classes):
        logs = log_likelihoods.copy()[c]
        prob = np.log(prior_probs[c]) + (logs * X_test[i]).sum()
        doc_probs.append(prob)
        print(f"Doc {10+(i//3)}{'ejs'[i%3]} class {c} prob {(logs * X_test[i]).sum():.0f} post {prob:.0f}")
    predicted_class = np.argmax(doc_probs)
    actual_class = y_test[i]
    confusion[predicted_class][actual_class] += 1
    predictions.append(predicted_class)

print(confusion)

    