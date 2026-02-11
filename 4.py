import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def A1(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    train_precision = precision_score(y_train, train_pred, average="weighted")
    train_recall = recall_score(y_train, train_pred, average="weighted")
    train_f1 = f1_score(y_train, train_pred, average="weighted")
    test_precision = precision_score(y_test, test_pred, average="weighted")
    test_recall = recall_score(y_test, test_pred, average="weighted")
    test_f1 = f1_score(y_test, test_pred, average="weighted")
    print("Train Confusion Matrix:\n", train_cm)
    print("Test Confusion Matrix:\n", test_cm)
    print("Train Precision:", train_precision)
    print("Train Recall:", train_recall)
    print("Train F1:", train_f1)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1:", test_f1)
    return model

def A2(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape)
    print("R2 Score:", r2)
    return mse, rmse, mape, r2

def A3(df):
    X = df[["0", "1"]].values
    y = df["LABEL"].values
    plt.scatter(X[y==0][:,0], X[y==0][:,1], color="blue", label="Class 0")
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color="red", label="Class 1")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Training Data")
    plt.legend()
    plt.show()
    return X, y

def A4(X, y):
    x_vals = np.arange(X[:,0].min(), X[:,0].max(), 10)
    y_vals = np.arange(X[:,1].min(), X[:,1].max(), 10)
    xx, yy = np.meshgrid(x_vals, y_vals)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    predictions = model.predict(test_points)
    plt.scatter(test_points[predictions==0][:,0], test_points[predictions==0][:,1],
                color="blue", s=10, label="Class 0")
    plt.scatter(test_points[predictions==1][:,0], test_points[predictions==1][:,1],
                color="red", s=10, label="Class 1")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("kNN Output (k=3)")
    plt.legend()
    plt.show()
    return predictions

def A5(X, y):
    x_vals = np.arange(X[:,0].min(), X[:,0].max(), 10)
    y_vals = np.arange(X[:,1].min(), X[:,1].max(), 10)
    xx, yy = np.meshgrid(x_vals, y_vals)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    for k in [1, 3, 5, 7, 9]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)
        predictions = model.predict(test_points)
        plt.figure()
        plt.scatter(test_points[predictions==0][:,0], test_points[predictions==0][:,1],
                    color="blue", s=10, label="Class 0")
        plt.scatter(test_points[predictions==1][:,0], test_points[predictions==1][:,1],
                    color="red", s=10, label="Class 1")
        plt.title(f"kNN Output (k = {k})")
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.legend()
        plt.show()

def A6(df):
    X = df[["0", "1"]].values
    y = df["LABEL"].values
    plt.scatter(X[y==0][:,0], X[y==0][:,1], color="blue", label="Class 0")
    plt.scatter(X[y==1][:,0], X[y==1][:,1], color="red", label="Class 1")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title("Project Dataset (2 Features)")
    plt.legend()
    plt.show()
    return X, y

def A7(X_train, y_train):
    params = {"n_neighbors": list(range(1, 21))}
    model = KNeighborsClassifier()
    grid = GridSearchCV(model, params, cv=5)
    grid.fit(X_train, y_train)
    rand = RandomizedSearchCV(model, params, n_iter=10, cv=5, random_state=1)
    rand.fit(X_train, y_train)
    print("Best k using GridSearch:", grid.best_params_)
    print("Best score using GridSearch:", grid.best_score_)
    print("Best k using RandomSearch:", rand.best_params_)
    print("Best score using RandomSearch:", rand.best_score_)
    return grid.best_params_, rand.best_params_

def main():
    df = pd.read_csv("dataset.csv")
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("A1")
    model = A1(X_train, X_test, y_train, y_test)
    print("\nA2")
    A2(X_train, X_test, y_train, y_test)
    print("\nA3")
    X_two, y_two = A3(df)
    print("\nA4")
    A4(X_two, y_two)
    print("\nA5")
    A5(X_two, y_two)
    print("\nA6")
    A6(df)
    print("\nA7")
    A7(X_train, y_train)

if __name__ == "__main__":
    main()
