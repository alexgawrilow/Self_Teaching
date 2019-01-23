import numpy as np


def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def rmse(y_true, y_pred):
    return np.sqrt(np.linalg.norm(y_true - y_pred) ** 2 / len(y_true))


class Network:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons  # list of number of neurons for each layer
        self.n_layers = len(n_neurons)  # number of layers in neural network

        # initialize weights and biases of neural network from gaussian distribution
        self.biases = [np.random.randn(i, 1) for i in n_neurons[1:]]
        self.weights = [
            np.random.randn(i, j) for i, j in zip(n_neurons[1:], n_neurons[:-1])
        ]

    def feedforward(self, x):
        """Returns the output of the network when x is fed into."""
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.dot(w, x) + b)
        return x

    def train(self, tr_data, epochs, batch_size, eta, test_data=None):
        """Train the neural network using stochastic gradient descent by approximating
        the gradient of the cost-function only using a batch_size of the training data.
        """
        for epoch in range(epochs):
            batches = Network.create_batches(tr_data, batch_size)
            for batch in batches:
                self.update_weights_biases(batch, eta)

            if test_data:
                print("Epoch {:2}: {:.3f}".format(epoch, self.evaluate(test_data)))

    @staticmethod
    def create_batches(tr_data, batch_size):
        """Creates a list of batches. Each batch contains `batch_size` samples from
        `tr_data`."""
        n_train = len(tr_data)  # number of training data
        np.random.shuffle(tr_data)
        return [tr_data[i : i + batch_size] for i in range(0, n_train, batch_size)]

    def update_weights_biases(self, batch, eta):
        """Updates the network's weights and biases using stochastic gradient descent.
        The gradient is approximated by using oa batch generated from training data.
        Weights and biases are updated by -eta * gradient."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        step_size = eta / len(batch)
        self.weights = [w - step_size * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - step_size * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple `(nabla_w, nabla_b)` representing the approximated gradient
        of the cost function. `nabla_w` and `nabla_b` are lists of numpy arrays similar
        to `self.weights` and `self.biases`."""
        nabla_w = [np.random.normal(scale=0.1, size=w.shape) for w in self.biases]
        nabla_b = [np.random.normal(scale=0.1, size=b.shape) for b in self.biases]

        return nabla_w, nabla_b

    def evaluate(self, test_data):
        """Returns the result of the current state of the neural network on test
        data."""
        x_test, y_test = [], []
        for x, y in test_data:
            x_test.append(x)
            y_test.append(y)

        y_pred = [self.feedforward(x) for x in x_test]
        return rmse(np.array(y_test), np.array(y_pred))
