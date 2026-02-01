import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def A1(df):
    A = df["0"]
    B = df["1"]
    dot = np.dot(A, B)
    normA = np.linalg.norm(A)
    normB = np.linalg.norm(B)
    return dot,normA,normB
def A2(df):
    class1 = df["0"]
    class2 = df["1"]
    mean1=np.mean(class1,axis=0)
    mean2=np.mean(class2,axis=0)
    std1=np.std(class1,axis=0)
    std2=np.std(class2,axis=0)
    distance=np.linalg.norm(mean1-mean2)
    return mean1,mean2,std1,std2,distance
def A3(df):
    data=df["0"]
    mean=np.mean(data)
    var=np.var(data)
    hist,bins=np.histogram(data,bins=10)
    print("mean:",mean)
    print("var:",var)
    plt.hist(data, bins=10)
    plt.xlabel("data values")
    plt.ylabel("Frequency")
    plt.title("Histogram of data")
    plt.show()
def minikowski_distance(feature1,feature2,p):
    sum=0
    for i in range(len(feature1)):
        sum+=abs(feature1[i]-feature2[i])**p
        return sum**(1/p)
def A4(df):
    feature1=df["0"]
    feature2=df["1"]
    distances=[]
    p_val=range(1,11)
    for p in p_val:
        d=minikowski_distance(feature1,feature2,p)
        distances.append(d)
    plt.plot(p_val,distances)
    plt.xlabel(" p values")
    plt.ylabel("minikowski_distance")
    plt.title("minikowski_distance vs p")
    plt.grid(True)
    plt.show()
def A5(df):
    vector1=df["0"]
    vector2=df["1"]
    p=10
    custom_dist = minikowski_distance(vector1, vector2, p)
    scipy_dist = minkowski(vector1, vector2, p)
    print("custom distance:",custom_dist)
    print("scipy distance:",scipy_dist)
def A6(df):
    X = df[['0','1']].values
    #dummy variables
    y = np.array([0 if i % 2 == 0 else 1 for i in range(len(df))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3    )
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    return X_train, X_test, y_train, y_test
def A7(X_train,y_train):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,y_train)	
    print("trained")
    return neigh
def A8(X_test, y_test, neigh):
    acc = neigh.score(X_test, y_test)
    print("Accuracy on test set:", acc)
    return acc

def A9(neigh, X_test):
    predictions = neigh.predict(X_test)
    print("Predictions for test set:", predictions)
    return predictions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def custom_knn(X_train, y_train, X_test, k=3):
    predictions = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            d = euclidean_distance(test_point, X_train[i])
            distances.append((d, y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]
        labels = [label for _, label in k_nearest]
        predictions.append(max(set(labels), key=labels.count))
    return np.array(predictions)
def A10(X_train, y_train, X_test):
    y_pred_custom = custom_knn(X_train, y_train, X_test, k=3)
    print("Custom kNN predictions:", y_pred_custom)
    return y_pred_custom
def A11(X_train, y_train, X_test, y_test):
    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        y_pred = custom_knn(X_train, y_train, X_test, k)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)
        print(f"k = {k}, Accuracy = {acc}")
    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy")
    plt.grid(True)
    plt.show()
def A12(y_true, y_pred):
    classes = np.unique(y_true)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print("Confusion Matrix:", cm)
    accuracy = np.trace(cm) / np.sum(cm)
    print("Accuracy:", accuracy)
    for i in range(len(classes)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        print(f"Class {i} -> Precision: {precision}, Recall: {recall}, F1-score: {f1}")
    return cm
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)
def precision_score(y_true, y_pred, cls):
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fp = np.sum((y_true != cls) & (y_pred == cls))
    return tp / (tp + fp) if (tp + fp) != 0 else 0
def recall_score(y_true, y_pred, cls):
    tp = np.sum((y_true == cls) & (y_pred == cls))
    fn = np.sum((y_true == cls) & (y_pred != cls))
    return tp / (tp + fn) if (tp + fn) != 0 else 0
def f1_score(y_true, y_pred, cls):
    p = precision_score(y_true, y_pred, cls)
    r = recall_score(y_true, y_pred, cls)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0
def A13(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    for cls in np.unique(y_true):
        print(f"Class {cls}:")
        print("Precision:", precision_score(y_true, y_pred, cls))
        print("Recall:", recall_score(y_true, y_pred, cls))
        print("F1-score:", f1_score(y_true, y_pred, cls))
def A14(X_train, y_train, X_test, y_test):
    y_pred = custom_knn(X_train, y_train, X_test, k=3)
    print("Custom kNN Accuracy:", accuracy_score(y_test, y_pred))
    A12(y_test, y_pred)
    A13(y_test, y_pred)
def main():
    df = pd.read_csv("dataset.csv")
    print("A1")
    dot,normA,normB=A1(df)
    print("Dot product:",dot)
    print("Norm A:",normA)
    print("Norm B:",normB)
    print("A2")
    mean1,mean2,std1,std2,distance=A2(df)
    print("mean1:",mean1)
    print("mean2:",mean2)
    print("std1:",std1)
    print("std2:",std2)
    print("distance:",distance)
    print("A3")
    A3(df)
    print("A4")
    A4(df)
    print("A5")
    A5(df)
    print("A6")
    X_train, X_test, y_train, y_test=A6(df)
    A6(df)
    print("A7")
    neigh=A7(X_train,y_train)
    print("A8")
    A8(X_train,y_train,neigh)
    print("A9")
    A9(neigh, X_test)
    print("A10")
    A10(X_train, y_train, X_test)
    print("A11")
    A11(X_train, y_train, X_test, y_test)
    print("A12")
    A12(y_true, y_pred)
    print("A13")
    A13(y_true, y_pred)
    print("A14")
    A14(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
