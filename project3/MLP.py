from __future__ import division

import numpy as np

import Assignment3.ActivationFunction as af

'''
# static function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def softmax_crossentropy(x, y):
    x_p = x[np.arange(len(x)), y]
    entropy = - x_p + np.log(np.sum(np.exp(x), axis=-1))
    return entropy
'''


def grad_softmax_crossentropy(x, y):
    ones_for_answers = np.zeros_like(x)
    ones_for_answers[np.arange(len(x)), y] = 1
    softmax = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    return (- ones_for_answers + softmax) / x.shape[0]





def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

class MLP:

    def __init__(self, activation, input_units, hidden_units, output_units, learning_rate):
        self.layers = []
        if len(hidden_units) == 0:
            self.layers.append(Layer(input_units, output_units,learning_rate))
            return
        self.layers.append(Layer(input_units, hidden_units[0], learning_rate))
        self.layers.append(af.get_activation_function(activation))
        for i, _ in enumerate(hidden_units):
            if i == len(hidden_units) - 1:
                break
            self.layers.append(Layer(hidden_units[i], hidden_units[i + 1], learning_rate))
            self.layers.append(af.get_activation_function(activation))
        self.layers.append(Layer(hidden_units[-1], output_units))
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

    def fit(self, x, y, epochs,batch_size, shuffle,lam):
        # input_units, output_units, learning_rate=0.1, epochs=25, dense_output_units=[100, 200], batchsize=32,
        # shuffle=True
        # if len(self.hidden_units) < 2:
        #    return
        for i in range(epochs):
            if i / epochs % 0.1 == 0:
                print("Process: %f" % (i / epochs))

            for x_batch, y_batch in iterate_minibatches(x, y, batchsize=batch_size, shuffle=shuffle):
                layer_activations = self.eval(x_batch)
                layer_inputs = [x_batch] + layer_activations
                logits = layer_activations[-1]
                # loss = softmax_crossentropy(logits, y_batch)
                loss_grad = grad_softmax_crossentropy(logits, y_batch)
                for layer_index in range(len(self.layers))[::-1]:
                    layer = self.layers[layer_index]
                    loss_grad = layer.back_prop(layer_inputs[layer_index], loss_grad,lam)

    def eval(self, x):
        activations = []
        input = x
        # Looping through each layer
        for l in self.layers:
            activations.append(l.eval(input))
            input = activations[-1]
        assert len(activations) == len(self.layers)
        return activations

    def predict(self, x):
        prediction = self.eval(x)[-1]
        return prediction.argmax(axis=-1)


class Layer:
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(2 / (input_units + output_units)),
                                        size=(input_units, output_units))
        self.biases = np.zeros(output_units)

    def eval(self, input):
        return np.dot(input, self.weights) + self.biases

    def back_prop(self, input, d_chain,lam):
        grad_input = np.dot(d_chain, self.weights.T)
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, d_chain)+lam*self.weights
        grad_biases = d_chain.mean(axis=0) * input.shape[0]
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input
