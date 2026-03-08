import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# A4 : Equal Width Binning
def equal_width_binning(data, bins=4):
    min_val = min(data)
    max_val = max(data)
    width = (max_val - min_val) / bins
    binned = []
    for value in data:
        bin_index = int((value - min_val) / width)
        if bin_index == bins:
            bin_index -= 1
        binned.append(bin_index)
    return binned

# A1 : Entropy Calculation
def entropy(data):
    total = len(data)
    counts = Counter(data)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# A2 : Gini Index
def gini_index(data):
    total = len(data)
    counts = Counter(data)
    gini = 1
    for count in counts.values():
        p = count / total
        gini -= p ** 2
    return gini

# Information Gain
def information_gain(feature, target):
    total_entropy = entropy(target)
    values = set(feature)
    weighted_entropy = 0
    for v in values:
        subset = [target[i] for i in range(len(target)) if feature[i] == v]
        weight = len(subset) / len(target)
        weighted_entropy += weight * entropy(subset)
    gain = total_entropy - weighted_entropy
    return gain

# A3 : Find Root Node
def find_best_feature(data, target):
    gains = []
    for col in data.columns:
        gain = information_gain(list(data[col]), target)
        gains.append(gain)
    best_index = np.argmax(gains)
    return data.columns[best_index]

# A5 : Simple Decision Tree
class Node:
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.children = {}
def build_tree(data, target):
    if len(set(target)) == 1:
        return Node(label=target[0])
    if len(data.columns) == 0:
        return Node(label=Counter(target).most_common(1)[0][0])
    best_feature = find_best_feature(data, target)
    node = Node(feature=best_feature)
    values = set(data[best_feature])
    for v in values:
        subset_index = [i for i in range(len(data)) if data[best_feature].iloc[i] == v]
        sub_data = data.iloc[subset_index].drop(columns=[best_feature])
        sub_target = [target[i] for i in subset_index]
        child = build_tree(sub_data, sub_target)
        node.children[v] = child
    return node

def print_tree(node, depth=0):
    if node.label is not None:
        print("  " * depth + "Label:", node.label)
        return
    print("  " * depth + "Feature:", node.feature)
    for value, child in node.children.items():
        print("  " * depth + "-->", value)
        print_tree(child, depth + 1)

iris = load_iris()
data = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
target = list(iris.target)
for col in data.columns:
    data[col] = equal_width_binning(data[col], bins=4)
print("Entropy of target:", entropy(target))
print("Gini Index:", gini_index(target))
root_feature = find_best_feature(data, target)
print("Best Root Feature:", root_feature)
tree = build_tree(data, target)
print("\nDecision Tree Structure")
print_tree(tree)
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()

# A7 Decision Boundary
X = iris.data[:, :2]
y = iris.target
model = DecisionTreeClassifier()
model.fit(X, y)
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.show()