import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('HW4/MNIST', train=True, download=True, transform=transform), batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('HW4/MNIST', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

d = 28 * 28 
d1 = 300
k = 10        

# W1 = np.random.rand(d1, d)*2 -1
# W2 = np.random.rand(k, d1)*2 -1

W1 = np.zeros((d1,d))
W2 = np.zeros((k,d1))

learning_rate = 0.1
num_epochs = 10

train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        x = data.numpy()
        y_true = F.one_hot(target, num_classes=k).numpy()

        z1 = np.dot(W1, x.T)
        a1 = 1 / (1 + np.exp(-z1)) 
        z2 = np.dot(W2, a1)
        y_hat = np.exp(z2 ) / np.sum(np.exp(z2), axis=0) 

        loss = -np.sum(y_true.T * np.log(y_hat)) / x.shape[0]

        grad_z2 = y_hat - y_true.T
        grad_W2 = np.dot(grad_z2, a1.T) / x.shape[0]
        grad_a1 = np.dot(W2.T, grad_z2)
        grad_z1 = grad_a1 * a1 * (1 - a1)  
        grad_W1 = np.dot(grad_z1, x) / x.shape[0]

        W2 -= learning_rate * grad_W2
        W1 -= learning_rate * grad_W1
        
        total_train_loss += loss
        
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Testing
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            x = data.numpy()
            y_true = F.one_hot(target, num_classes=k).numpy()

            z1 = np.dot(W1, x.T)
            a1 = 1 / (1 + np.exp(-z1))
            z2 = np.dot(W2, a1)
            y_hat = np.exp(z2 - np.max(z2, axis=0)) / np.sum(np.exp(z2 - np.max(z2, axis=0)), axis=0)

            test_loss = -np.sum(y_true.T * np.log(y_hat)) / x.shape[0]
            total_test_loss += test_loss

            _, predicted = torch.max(torch.from_numpy(y_hat.T), 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Training Loss: {avg_train_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, '
          f'Test Accuracy: {test_accuracy:.2f}%')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

