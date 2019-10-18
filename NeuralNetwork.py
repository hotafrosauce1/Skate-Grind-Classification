import numpy as np
import keras
from keras.utils import np_utils
from sklearn.preprocessing import scale
import json
import random
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

class Regularization(object):
    @staticmethod
    def get(regularizaton_method):
        return reg_methods[regularizaton_method]

    @staticmethod
    def L1(weights, lmbda):
        return lmbda * np.nan_to_num(np.sign(weights))

    @staticmethod
    def L2(weights, lmbda):
        return lmbda * weights

    @staticmethod
    def none(weights, lmbda):
        return 0

class CrossEntropy(object):

    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return output_activation - y

class QuadraticError(object):

    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return (output_activation - y) * sigmoid_prime(weighted_input)


class Network(object):

    reg_methods = {'L1': Regularization.L1, 'L2': Regularization.L2, 'None': Regularization.none}
    cost_functions = {'cross-entropy': CrossEntropy.output_error, 'mse': QuadraticError.output_error}

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.bias = []
        self.weights = []
        self.layers = layers
        self.num_classes = layers[-1]
        self.class_map = {}
        self.inverse_map = {}

        prev_layer_neurons = None
        for (layer_index, neurons) in enumerate(layers):
            if layer_index == 0:
                self.weights.append(np.random.rand(neurons))
                self.bias.append(np.zeros(neurons))
                prev_layer_neurons = neurons
            else:
                self.weights.append(np.array([np.random.randn(prev_layer_neurons) / np.sqrt(prev_layer_neurons) for _ in range(neurons)]))
                self.bias.append(np.random.rand(neurons))
                prev_layer_neurons = neurons

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

    def save(self, filename):
        """Save the neural network to the file ‘‘filename‘‘."""
        data = {"weights": [w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.bias]}
        f = open(filename , "w")
        json.dump(data, f)
        f.close()

    def predict(self, x):
        output_activation = self.feedforward(x, is_prediction = True)
        prediction_index = np.argmax(output_activation)
        return self.class_map[prediction_index], output_activation

    def evaluate(self, X_data, y_data, X_train, y_train):
        val_size = len(X_data)
        train_size = len(X_train)
        preds_val = [1 if (self.predict(X_data[i])[0] == y_data[i]) else 0 for i in range(val_size)]
        preds_train = [1 if (self.predict(X_train[i])[0] == y_train[i]) else 0 for i in range(train_size)]
        val_accuracy = round(sum(preds_val) / val_size * 100, 2)
        train_accuracy = round(sum(preds_train) / train_size * 100, 2)
        return "Validation Accuracy: {}%, Training Accuracy: {}%".format(val_accuracy, train_accuracy)

    def train(self, X_train, y_train, X_validate = None, y_validate = None,
              lr = 3, batch_size = 100, epochs = 100, regularizaton_method = 'None',
              lmbda = 0.1, cost_function = 'cross-entropy'):
        assert len(X_train[0]) == self.layers[0]

        n = len(X_train)

        self.create_class_mapping(y_train)

        X_len = len(X_train)
        true_batch = min(batch_size, X_len)
        if true_batch < 100:
            batch_size = true_batch

        for epoch in range(epochs):

            X_and_y = np.c_[X_train, y_train]
            np.random.shuffle(X_and_y)

            mini_batches = [X_and_y[i:i + batch_size] for i in range(0, len(X_and_y), batch_size)]

            for batch in mini_batches:
                for train_example in batch:
                    x_train_example = train_example[:-1]
                    y_train_example = train_example[-1]
                    mapped_y = self.inverse_map[y_train_example]
                    categorized_y = np_utils.to_categorical(mapped_y, num_classes = self.num_classes)

                    activations, weighted_inputs = self.feedforward(x_train_example)
                    bias_gradients, weight_gradients = self.backpropagation(categorized_y, activations, weighted_inputs,
                                                                            cost_function)

                    self.update_weights_and_biases(bias_gradients, weight_gradients,
                                                   lr, batch_size, regularizaton_method, lmbda, n)

            if (X_validate is None) or (y_validate is None):
                print("Epoch {}".format(epoch + 1))
            else:
                print("Epoch: {}, {}".format(epoch + 1, self.evaluate(X_validate, y_validate, X_train, y_train)))

    def feedforward(self, input, is_prediction = False):
        activations = []
        weighted_inputs = []
        input = list(map(lambda x: int(x), input))

        for (layer_index, neurons) in enumerate(self.layers):
            if layer_index == 0:
                weighted_input = self.weights[0] * input
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)
            else:
                weighted_input = np.dot(self.weights[layer_index], activations[layer_index - 1]) + self.bias[layer_index]
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)

        activations = np.array(activations)
        weighted_inputs = np.array(weighted_inputs)

        if is_prediction:
            return activations[-1]
        else:
            return activations, weighted_inputs

    def backpropagation(self, y, activations, weighted_inputs, cost_function):
        bias_grad = {}
        weight_grad = {}
        num_layers = len(self.layers)

        for layer_index in range(num_layers - 1, 0, -1):
            if layer_index == num_layers - 1:
                output_layer_error =  self.cost_functions[cost_function](activations[layer_index], y, weighted_inputs[layer_index])
                bias_grad[layer_index] = output_layer_error
                weight_grad[layer_index] = np.outer(output_layer_error, activations[layer_index - 1])
            else:
                next_layer_weights_t = self.weights[layer_index + 1].T
                next_layer_errors = bias_grad[layer_index + 1]
                curr_layer_error = np.dot(next_layer_weights_t, next_layer_errors) * sigmoid_prime(weighted_inputs[layer_index])
                bias_grad[layer_index] = curr_layer_error
                weight_grad[layer_index] = np.outer(curr_layer_error, activations[layer_index - 1])

        return bias_grad, weight_grad

    def update_weights_and_biases(self, bias_gradients, weight_gradients, lr, batch_size,
                                  regularizaton_method, lmbda, n):
        reg_method = self.reg_methods[regularizaton_method]
        for layer_index in range(1, len(self.layers)):
            reg = reg_method(self.weights[layer_index], lmbda)
            self.bias[layer_index] -= (lr / batch_size * bias_gradients[layer_index])
            self.weights[layer_index] -= (lr / batch_size * weight_gradients[layer_index]) - (lr / n * reg)


    def create_class_mapping(self, y):
        class_map = {}
        inverse_map = {}
        curr_class_num = 0
        for val in np.unique(y):
            if val in class_map.values() and val in inverse_map.keys():
                continue
            else:
                class_map[curr_class_num] = val
                inverse_map[val] = curr_class_num
                curr_class_num += 1
        self.class_map = class_map
        self.inverse_map = inverse_map
#
# n = Network([784,100,10])
# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train = list(map(lambda x: x.reshape((784,)), X_train))
# X_train = scale(X_train)
#
# X_test = list(map(lambda x: x.reshape((784,)), X_test))
# X_test = scale(X_test)
#
# n.train(X_train, y_train, X_validate = X_test, y_validate = y_test,
#  batch_size = 10, epochs = 10, lr = 0.25, cost_function = 'cross-entropy',
#   regularizaton_method = 'L2', lmbda = 0.1)
#
# n.save('DigitClassifier.json')
