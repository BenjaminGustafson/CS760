import numpy as np
import matplotlib.pyplot as plt

means = np.array([[-1, -1], [1, -1], [0,1]])
covs = np.array([ [[2,0.5],[0.5,1]],  [[1,-0.5],[-0.5,2]], [[1,0],[0,2]]])
sigmas = [0.5,1,2,4,8]

num_samples = 100
kmeans_objective = []
gmm_objective = []
kmeans_accuracy = []
gmm_accuracy = []

K = 3

for sigma in sigmas:
    # Set up dataset
    samples1 =  np.random.multivariate_normal(means[0],sigma*covs[0], 100)
    samples2 =  np.random.multivariate_normal(means[1],sigma*covs[1], 100)
    samples3 =  np.random.multivariate_normal(means[2],sigma*covs[2], 100)
    X = np.vstack((samples1,samples2,samples3))
    y = np.array(np.concatenate(([0]*num_samples,[1]*num_samples,[2]*num_samples)))
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    #K-means
    random_indices = np.random.choice(X.shape[0], K, replace=False)
    centers = X[random_indices]

    prev = np.array([])
    while not np.array_equal(prev,centers):
        prev = centers
        closest_centers = []
        dists_to_closest = []
        center_counts = np.zeros(K)
        centers = np.zeros((K,2))
        for point in X:
            dist_to_center = [np.sum((point - center)**2) for center in prev]
            dists_to_closest.append(np.min(dist_to_center))
            center_i = np.argmin(dist_to_center)
            closest_centers.append(center_i)
            center_counts[center_i] += 1
            centers[center_i] += point
        centers = (centers.T/center_counts).T
    
    mean_to_label = []
    for m in means:
        dists = [np.sum((m - c)**2) for c in centers]
        mean_to_label.append(np.argmin(dists))
    mean_to_label = np.array(mean_to_label)


    # Evaluation
    accuracy = (y == mean_to_label[closest_centers]).sum() / len(y)
    kmeans_accuracy.append(accuracy)

    kmeans_objective.append(np.array(dists_to_closest).sum())

    #Plot k-means
    # colors = ['r', 'g', 'b']
    # for i in range(len(X)):
    #     plt.scatter(X[i, 0], X[i, 1], color=colors[closest_centers[i]], marker='o')
    # centers = np.array(centers)
    # plt.scatter(centers[:, 0], centers[:, 1], color='k', marker='x', label='Centers')
    # plt.scatter(means[:, 0], means[:, 1], color='k', marker='+', label='Centers')
    # plt.show()

    # GMM


plt.plot(sigmas, kmeans_objective, label="K-means objective")
plt.xticks(sigmas)
plt.show()
plt.plot(sigmas, kmeans_accuracy, label="K-means accuracy")
plt.xticks(sigmas)
plt.show()







        



    