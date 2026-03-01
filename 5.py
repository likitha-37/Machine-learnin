import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def A1(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    return model, y_train_pred


def A2(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    test_r2 = r2_score(y_test, y_test_pred)
    print(" Train Metrics ")
    print("MSE:", train_mse)
    print("RMSE:", train_rmse)
    print("MAPE:", train_mape)
    print("R2:", train_r2)
    print(" Test Metrics ")
    print("MSE:", test_mse)
    print("RMSE:", test_rmse)
    print("MAPE:", test_mape)
    print("R2:", test_r2)

def A3(df):
    X = df.drop("0", axis=1)
    y = df["0"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Regression with multiple attributes")
    A2(model, X_train, y_train, X_test, y_test)

def A4(X_train):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(X_train)
    print("Cluster Labels:", kmeans.labels_)
    print("Cluster Centers:\n", kmeans.cluster_centers_)
    return kmeans

def A5(X_train, kmeans):
    sil = silhouette_score(X_train, kmeans.labels_)
    ch = calinski_harabasz_score(X_train, kmeans.labels_)
    db = davies_bouldin_score(X_train, kmeans.labels_)
    print("Silhouette Score:", sil)
    print("Calinski-Harabasz Score:", ch)
    print("Davies-Bouldin Index:", db)

def A6(X_train):
    k_values = range(2, 11)
    sil_scores = []
    ch_scores = []
    db_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train)
        sil_scores.append(silhouette_score(X_train, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(X_train, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X_train, kmeans.labels_))
    plt.plot(k_values, sil_scores, label="Silhouette")
    plt.plot(k_values, ch_scores, label="CH Score")
    plt.plot(k_values, db_scores, label="DB Index")
    plt.xlabel("k value")
    plt.ylabel("Score")
    plt.title("Clustering Scores vs k")
    plt.legend()
    plt.show()

def A7(X_train):
    distortions = []
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train)
        distortions.append(kmeans.inertia_)
    plt.plot(range(2, 20), distortions)
    plt.xlabel("k value")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    plt.show()

def main():
    df = pd.read_csv("dataset.csv")
    X = df[["1"]]     
    y = df["0"]       
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("A1")
    model, y_train_pred = A1(X_train, y_train)
    print("\nA2")
    A2(model, X_train, y_train, X_test, y_test)
    print("\nA3")
    A3(df)
    print("\nA4")
    X_cluster = df.drop("LABEL", axis=1)
    X_train_cluster, _, _, _ = train_test_split(X_cluster, X_cluster, test_size=0.3)
    kmeans = A4(X_train_cluster)
    print("\nA5")
    A5(X_train_cluster, kmeans)
    print("\nA6")
    A6(X_train_cluster)
    print("\nA7")
    A7(X_train_cluster)

if __name__ == "__main__":
    main()