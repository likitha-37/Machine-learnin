import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def load_data():
    df = pd.read_csv("dataset.csv")   
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    return X, y

def tune_model(model, params, X_train, y_train):
    search = RandomizedSearchCV(model, params, n_iter=5, cv=3)
    search.fit(X_train, y_train)
    return search.best_estimator_

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return acc, prec, rec, f1

def run_models(X_train, X_test, y_train, y_test):
    results = []
    models = {
        "SVM": (SVC(), {"C": [0.1, 1, 10]}),
        "DecisionTree": (DecisionTreeClassifier(), {"max_depth": [3, 5, 10]}),
        "RandomForest": (RandomForestClassifier(), {"n_estimators": [50, 100]}),
        "AdaBoost": (AdaBoostClassifier(), {"n_estimators": [50, 100]}),
        "NaiveBayes": (GaussianNB(), {}),
        "MLP": (MLPClassifier(max_iter=500), {"hidden_layer_sizes": [(50,), (100,)]})
    }
    for name, (model, params) in models.items():
        print("\nRunning:", name)
        if params:
            model = tune_model(model, params, X_train, y_train)
        else:
            model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        train_acc, train_prec, train_rec, train_f1 = get_metrics(y_train, y_train_pred)
        y_test_pred = model.predict(X_test)
        test_acc, test_prec, test_rec, test_f1 = get_metrics(y_test, y_test_pred)
        results.append([
            name,
            train_acc, test_acc,
            train_prec, test_prec,
            train_rec, test_rec,
            train_f1, test_f1
        ])

    return results

def print_results(results):
    columns = [
        "Model",
        "Train Acc", "Test Acc",
        "Train Prec", "Test Prec",
        "Train Recall", "Test Recall",
        "Train F1", "Test F1"
    ]
    df = pd.DataFrame(results, columns=columns)
    print("\nFINAL RESULT TABLE:\n")
    print(df)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    results = run_models(X_train, X_test, y_train, y_test)
    print_results(results)

if __name__ == "__main__":
    main()