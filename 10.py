import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from lime.lime_tabular import LimeTabularExplainer
import shap


# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv("dataset.csv")   # 🔴 change if needed
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    return X, y


# ---------------- A1 : CORRELATION ----------------
def correlation_heatmap(X):

    corr = X.corr()

    plt.figure()
    sns.heatmap(corr, annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()


# ---------------- TRAIN MODEL ----------------
def train_model(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, model


# ---------------- A2 : PCA 99% ----------------
def pca_99(X_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    var = np.cumsum(pca.explained_variance_ratio_)

    k = np.argmax(var >= 0.99) + 1

    print("Components for 99% variance:", k)

    pca = PCA(n_components=k)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


# ---------------- A3 : PCA 95% ----------------
def pca_95(X_train, X_test):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)

    var = np.cumsum(pca.explained_variance_ratio_)

    k = np.argmax(var >= 0.95) + 1

    print("Components for 95% variance:", k)

    pca = PCA(n_components=k)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


# ---------------- A4 : FEATURE SELECTION ----------------
def feature_selection(X_train, X_test, y_train):

    model = RandomForestClassifier()

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=3,
        direction="forward"
    )

    sfs.fit(X_train, y_train)

    selected = sfs.get_support()

    print("Selected Features:", X_train.columns[selected])

    X_train = X_train.loc[:, selected]
    X_test = X_test.loc[:, selected]

    return X_train, X_test


# ---------------- A5 : LIME ----------------
def lime_explain(model, X_train, X_test):

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=["0","1"],
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        model.predict_proba
    )

    print("\nLIME Explanation:")
    print(exp.as_list())


# ---------------- A5 : SHAP ----------------
def shap_explain(model, X_train):

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    print("\nSHAP Summary Plot")
    shap.summary_plot(shap_values, X_train)


# ---------------- MAIN ----------------
def main():

    X, y = load_data()

    correlation_heatmap(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Original
    acc_orig, model = train_model(X_train, X_test, y_train, y_test)
    print("\nOriginal Accuracy:", acc_orig)

    # A2 (99%)
    X_train_pca, X_test_pca = pca_99(X_train, X_test)
    acc_99, _ = train_model(X_train_pca, X_test_pca, y_train, y_test)
    print("Accuracy with PCA 99%:", acc_99)

    # A3 (95%)
    X_train_pca, X_test_pca = pca_95(X_train, X_test)
    acc_95, _ = train_model(X_train_pca, X_test_pca, y_train, y_test)
    print("Accuracy with PCA 95%:", acc_95)

    # A4 (Feature Selection)
    X_train_fs, X_test_fs = feature_selection(X_train, X_test, y_train)
    acc_fs, model_fs = train_model(X_train_fs, X_test_fs, y_train, y_test)
    print("Accuracy with Feature Selection:", acc_fs)

    # A5
    lime_explain(model_fs, X_train_fs, X_test_fs)
    shap_explain(model_fs, X_train_fs)


if __name__ == "__main__":
    main()