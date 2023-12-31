import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeClassifier

def read_data(filename):
    X = []
    y = []
    with open(filename, 'r') as file:
        for line in file:
            x1, x2, label = map(float, line.strip().split())
            X.append((x1, x2))
            y.append(label)
    return (np.array(X),np.array(y))

class DecisionTree:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self._fit(X,y)

    def _fit(self, X, y):          
        best_split = self._find_best_split(X,y)        

        if best_split is None:
            # This occurs if X is empty (in that case we have nothing to loop over), 
            # or if all info gain ratios are 0 
            #  (in this case info_gain_ratio > best_info_gain_ratio is never true),
            # or if the entropy of any candidate splits is zero?
            # I don't know what the 3rd condition means
            majority_class = 1 if y.sum() >= len(y) / 2 else 0
            # note that majority_class is 1 in the case of a tie
            return {'leaf': majority_class}
        
        left_mask = X[:, best_split['feature']] >= best_split['threshold']
        right_mask = ~left_mask
        left_tree = self._fit(X[left_mask], y[left_mask])
        right_tree = self._fit(X[right_mask], y[right_mask])
        
        return {'node': best_split, 'left': left_tree, 'right': right_tree}


    def _find_best_split(self, X, y):
        best_split = None
        best_info_gain_ratio = 0

        num_samples, num_features = X.shape
        parent_entropy = self._entropy(y)

        for feature in range(num_features):
            for threshold in X[:, feature]:
                left_mask = X[:, feature] >= threshold
                right_mask = ~left_mask
        
                left = y[left_mask]
                right = y[right_mask]

                info_gain_ratio = self._info_gain_ratio(left, right, parent_entropy)

                if info_gain_ratio is None:
                    return None

                if info_gain_ratio > best_info_gain_ratio:
                    best_info_gain_ratio = info_gain_ratio
                    best_split = {'feature' : feature, 'threshold': threshold}
        
        return best_split

    def _info_gain_ratio(self, left, right, parent_entropy):
        
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)
        
        left_portion = len(left) / (len(left) + len(right))
        right_portion = 1 - left_portion

        info_gain = parent_entropy - ( left_portion * left_entropy) \
                    - (right_portion * right_entropy)
        

        if left_portion == 0 or right_portion == 0:
            return 0
        
        split_entropy = - (left_portion * np.log2(left_portion) \
                           + right_portion * np.log2(right_portion))

        if split_entropy == 0:
            return None


        info_gain_ratio = info_gain / split_entropy
        return info_gain_ratio

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        
        p1 = sum(y)/len(y)
        p0 = 1-p1

        if p1 == 0 or p0 == 0:
            return 0
        
        entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))
        return entropy
    

    def predict(self, X):
        predictions = []
        for sample in X:
            label = self._predict(self.tree, sample)
            predictions.append(label)
        return predictions
    
    def _predict(self, node, sample):
        if 'leaf' in node:
            return node['leaf']
        
        feature_index = node['node']['feature']
        threshold = node['node']['threshold']

        if sample[feature_index] >= threshold:
            return self._predict(node['left'], sample)
        else:
            return self._predict(node['right'], sample)
        
    def print(self):
        self._print_helper(self.tree)
    
    def _print_helper(self, node,depth=0):
        if 'leaf' in node:
            print('|\t'*depth + str(node['leaf']))
        else:
            print('|\t'*depth + f"If feature {node['node']['feature']+1} >=  {node['node']['threshold']}")
            self._print_helper(node['left'],depth+1)
            print('|\t'*depth + 'else')
            self._print_helper(node['right'],depth+1)

    def count_nodes(self):
        return self._count_helper(self.tree)
    
    def _count_helper(self, node):
        if 'leaf' in node:
            return 1
        else:
            return self._count_helper(node['left']) + self._count_helper(node['right'])

    def visualize(self):
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        #plt.scatter(X_class0[:, 0], X_class0[:, 1], c='b', marker='o', label='0')
        #plt.scatter(X_class1[:, 0], X_class1[:, 1], c='r', marker='x', label='1')
        ax = plt.gca()
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        #plt.legend(loc='best')
        plt.title('')
        min0 = min(X[:,0])
        min1 = min(X[:,1])
        max0 = max(X[:,0])
        max1 = max(X[:,1])
        self._visualize_helper(self.tree, min0, min1, max0, max1)
        plt.show()
    
    def _visualize_helper(self, node, min0, min1, max0, max1):
        if 'leaf' in node:
            edgecolor = 'r' if node['leaf'] == 1 else 'b'
            facecolor = 'red' if node['leaf'] == 1 else 'blue'
            rect = patches.Rectangle((min0, min1), max0 - min0, max1 - min1, linewidth=1, edgecolor=edgecolor, facecolor=facecolor, alpha=0.3)
            plt.gca().add_patch(rect)
        else:
            if node['node']['feature'] == 0:
                self._visualize_helper(node['right'], min0,min1, node['node']['threshold'], max1)
                self._visualize_helper(node['left'], node['node']['threshold'], min1, max0, max1)
            else:
                self._visualize_helper(node['right'], min0,min1,max0 , node['node']['threshold'])
                self._visualize_helper(node['left'], min0, node['node']['threshold'], max0, max1)


def accuracy(y, y_pred):
    return np.sum(y == y_pred)/len(y)

def plot(X,y):
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    plt.scatter(X_class0[:, 0], X_class0[:, 1], c='b', marker='o', label='0')
    plt.scatter(X_class1[:, 0], X_class1[:, 1], c='r', marker='x', label='1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.title('')
    plt.show()



tree = DecisionTree()
"""
print("Testing entropy ")
print(tree._entropy(np.array([0,1,1,0])))
print(tree._entropy(np.array([1,1,1])))
print(tree._entropy(np.array([])))
print(tree._entropy(np.array([1,1,1,0])))
print("Testing info gain ratio")
# https://en.wikipedia.org/wiki/Information_gain_ratio wind table, humidity table
print(tree._info_gain_ratio(np.array([1,1,1,1,1,1,0,0]), np.array([1,1,1,0,0,0]), 0.94 )) 
print(tree._info_gain_ratio(np.array([1,1,1,0,0,0,0]), np.array([1,1,1,1,1,1,0]), 0.94 ))
print(tree._info_gain_ratio(np.array([0,1,1,0]), np.array([]), 1.0 ))
print("Testing find best split")
print(tree._find_best_split(np.array([[0.5,0.5]]) , np.array([1])))
print(tree._find_best_split(np.array([[0.5,0.1],[0.2,0.6],[0.3,0.4]]) , np.array([1,1,1])))
"""

# (X,y) = read_data('HW2/data/Dbig.txt')
# # X = np.array([[0,0],[0,1],[1,0],[1,1]])
# # y = np.array([1,0,0,1])
# tree.fit(X, y)
# print("")
# tree.print()
# y_pred = tree.predict(X)
# tree.visualize()
#print(f"accuracy {accuracy(y,y_pred)}")
#plot(X,y)

(X,y) = read_data('HW2/data/Dbig.txt')
permuted_indices = np.random.permutation(X.shape[0])
perm_X = X[permuted_indices]
perm_y = y[permuted_indices]
X_test = perm_X[8192:]
y_test = perm_y[8192:]
for n in [32,128,2048,8192]:
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(perm_X[:n], perm_y[:n])
    num_nodes = clf.tree_.node_count
    y_pred = clf.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)

    # tree.fit(perm_X[:n], perm_y[:n])
    # y_pred = tree.predict(X_test)
    print(f"D{n} Accuracy {accuracy(y_test, y_pred)} num nodes {num_nodes}")
    # tree.visualize()
#  128 512 2048 8192
#n, error, num nodes, plot n vs err (learning curve)
# visualize boundary
# y_pred = tree.predict(X)