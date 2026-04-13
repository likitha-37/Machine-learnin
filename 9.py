import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# A1
def summation(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def error(target, output):
    return target - output


# A2 
def train_and_gate(activation_func, lr):
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    w = np.array([0.2, -0.75])
    b = 10
    epochs = 0
    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            net = summation(X[i], w, b)
            out = activation_func(net)
            err = y[i] - out
            total_error += err**2
            w = w + lr * err * X[i]
            b = b + lr * err
        epochs += 1
        if total_error <= 0.002:
            break
    return epochs

# A3
def compare_activation():
    print("Step Function")
    e1 = train_and_gate(step, 0.05)
    print("\nBipolar Step")
    e2 = train_and_gate(bipolar_step, 0.05)
    print("\nSigmoid")
    e3 = train_and_gate(sigmoid, 0.05)
    print("\nReLU")
    e4 = train_and_gate(relu, 0.05)
    print("\nIterations Comparison:")
    print("Step:", e1)
    print("Bipolar:", e2)
    print("Sigmoid:", e3)
    print("ReLU:", e4)

# A4 
def learning_rate_analysis():
    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    epochs_list = []
    for lr in rates:
        epochs = train_and_gate(step, lr)
        epochs_list.append(epochs)
    plt.plot(rates, epochs_list, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge")
    plt.title("Learning Rate vs Epochs")
    plt.grid(True)
    plt.show()

# A5 
def train_xor_gate(activation_func, lr):
    X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])
    y = np.array([0,1,1,0])
    w = np.array([0.2, -0.75])
    b = 10
    errors = []
    epochs = 0
    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            net = summation(X[i], w, b)
            out = activation_func(net)
            err = error(y[i], out)
            total_error += err**2
            w = w + lr * err * X[i]
            b = b + lr * err
        errors.append(total_error)
        epochs += 1
    print("Final Weights:", w, "Bias:", b)
    print("Epochs:", epochs)
    print("Final Error:", total_error)
    plt.plot(errors)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("XOR Error vs Epochs")
    plt.show()
    return epochs

def error(y, y_pred):
    return y - y_pred


# A6 
def customer_perceptron():
    X = np.array([
        [20,6,2,386],
        [16,3,6,289],
        [27,6,2,393],
        [19,1,2,110],
        [24,4,2,280],
        [22,1,5,167],
        [15,4,2,271],
        [18,4,2,274],
        [21,1,4,148],
        [16,2,4,198]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])
    X = X / np.max(X, axis=0)
    w = np.random.rand(4)
    b = np.random.rand()
    lr = 0.1
    epochs = 1000
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            net = np.dot(X[i], w) + b
            out = sigmoid(net)
            err = error(y[i], out)
            total_error += err**2
            w = w + lr * err * X[i]
            b = b + lr * err
        if epoch % 100 == 0:
            print("Epoch:", epoch, "Error:", total_error)
        if total_error <= 0.002:
            print("Converged at epoch:", epoch)
            break
    print("\nFinal Weights:", w)
    print("Final Bias:", b)
    print("\nPredictions:")
    for i in range(len(X)):
        net = np.dot(X[i], w) + b
        out = sigmoid(net)
        pred = 1 if out >= 0.5 else 0
        print("Customer", i+1, "->", pred, "(Actual:", y[i], ")")


def pseudo_inverse_method():
    X = np.array([
        [20,6,2,386],
        [16,3,6,289],
        [27,6,2,393],
        [19,1,2,110],
        [24,4,2,280],
        [22,1,5,167],
        [15,4,2,271],
        [18,4,2,274],
        [21,1,4,148],
        [16,2,4,198]
    ])
    y = np.array([1,1,1,0,1,0,1,1,0,0])
    X = X / np.max(X, axis=0)
    X_bias = np.c_[np.ones(len(X)), X]
    X_pinv = np.linalg.pinv(X_bias)
    weights = np.dot(X_pinv, y)
    print("Weights (Pseudo-Inverse):", weights)
    print("\nPredictions using Pseudo-Inverse:")
    for i in range(len(X)):
        net = np.dot(X_bias[i], weights)
        out = sigmoid(net)
        pred = 1 if out >= 0.5 else 0
        print("Customer", i+1, "->", pred, "(Actual:", y[i], ")")
def compare_results():
    print("\nPerceptron Results (from A6):")
    print("→ Learned iteratively using gradient updates")
    print("\nPseudo-Inverse Results (A7):")
    pseudo_inverse_method()
    print("\nObservation:")
    print("- Perceptron learns step-by-step (iterative)")
    print("- Pseudo-inverse gives direct solution (one step)")
    print("- Perceptron may take many epochs")
    print("- Pseudo-inverse is faster but sensitive to data")

def sigmoid_derivative(x):
    return x * (1 - x)

def backprop_and():
    print("\nA8 - Backpropagation AND Gate")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[0],[0],[1]])
    w_input_hidden = np.random.rand(2,2)
    w_hidden_output = np.random.rand(2,1)
    lr = 0.05
    for epoch in range(1000):
        hidden = sigmoid(np.dot(X, w_input_hidden))
        output = sigmoid(np.dot(hidden, w_hidden_output))
        error = y - output
        if np.mean(error**2) <= 0.002:
            print("Converged at epoch:", epoch)
            break
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(w_hidden_output.T) * sigmoid_derivative(hidden)
        w_hidden_output += hidden.T.dot(d_output) * lr
        w_input_hidden += X.T.dot(d_hidden) * lr
    print("Final Output:\n", output)


#  A9 
def backprop_xor():
    print("\nA9 - Backpropagation XOR Gate")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    w_input_hidden = np.random.rand(2,2)
    w_hidden_output = np.random.rand(2,1)
    lr = 0.05
    for epoch in range(1000):
        hidden = sigmoid(np.dot(X, w_input_hidden))
        output = sigmoid(np.dot(hidden, w_hidden_output))
        error = y - output
        if np.mean(error**2) <= 0.002:
            print("Converged at epoch:", epoch)
            break
        d_output = error * sigmoid_derivative(output)
        d_hidden = d_output.dot(w_hidden_output.T) * sigmoid_derivative(hidden)
        w_hidden_output += hidden.T.dot(d_output) * lr
        w_input_hidden += X.T.dot(d_hidden) * lr
    print("Final Output:\n", output)


#  A10 
def two_output_nodes():
    print("\nA10 - Two Output Nodes")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[1,0],[1,0],[1,0],[0,1]])
    w = np.random.rand(2,2)
    lr = 0.05
    for epoch in range(1000):
        output = sigmoid(np.dot(X, w))
        error = y - output
        if np.mean(error**2) <= 0.002:
            print("Converged at epoch:", epoch)
            break
        d_output = error * sigmoid_derivative(output)
        w += X.T.dot(d_output) * lr
    print("Final Output:\n", output)

# A11 
def mlp_and_xor():
    print("\nA11 - MLPClassifier")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y_and = np.array([0,0,0,1])
    model_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    model_and.fit(X, y_and)
    print("AND Predictions:", model_and.predict(X))
    y_xor = np.array([0,1,1,0])
    model_xor = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    model_xor.fit(X, y_xor)
    print("XOR Predictions:", model_xor.predict(X))


# A12 
def mlp_on_dataset():
    df = pd.read_csv("dataset.csv")  
    X = df.drop("LABEL", axis=1)
    y = df["LABEL"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy on Dataset:", acc)


#  MAIN 
def main():

    print("A2 - AND Gate using Step Function")
    train_and_gate(step, 0.05)

    print("\nA3 - Activation Comparison")
    compare_activation()

    print("\nA4 - Learning Rate Analysis")
    learning_rate_analysis()

    print("\nA5 - XOR Gate")
    train_xor_gate(step, 0.05)
    print("A6 - Customer Perceptron")
    customer_perceptron()
    print("A7 - Comparison with Pseudo-Inverse")
    compare_results()

    backprop_and()
    backprop_xor()
    two_output_nodes()
    mlp_and_xor()
    mlp_on_dataset()


if __name__ == "__main__":
    main()