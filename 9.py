import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from lime.lime_tabular import LimeTabularExplainer


# ---------------- LOAD DATA ----------------
def load_data():
    df = pd.read_csv("dataset.csv")   # 🔴 change file name if needed
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    return X, y


# ---------------- A1 : STACKING ----------------
def stacking_model():

    # Base models
    base_models = [
        ("svm", SVC(probability=True)),
        ("dt", DecisionTreeClassifier()),
        ("rf", RandomForestClassifier())
    ]

    # Meta model
    final_model = LogisticRegression()

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=final_model
    )

    return stack


# ---------------- A2 : PIPELINE ----------------
def build_pipeline(model):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    return pipe


# ---------------- A3 : LIME ----------------
def explain_model(pipe, X_train, X_test):

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["0", "1"],
        mode="classification"
    )

    # Explain one sample
    sample = X_test.iloc[0]

    exp = explainer.explain_instance(
        sample.values,
        pipe.predict_proba
    )

    print("\nLIME Explanation for first test sample:\n")
    print(exp.as_list())


# ---------------- MAIN ----------------
def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # A1
    print("Building Stacking Model...")
    stack = stacking_model()

    # A2
    print("Building Pipeline...")
    pipe = build_pipeline(stack)

    # Train
    pipe.fit(X_train, y_train)

    # Accuracy
    acc = pipe.score(X_test, y_test)
    print("\nTest Accuracy:", acc)

    # A3
    explain_model(pipe, X_train, X_test)


if __name__ == "__main__":
    main()