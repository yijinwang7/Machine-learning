import numpy as np
from enum import Enum, unique

@unique
class ActivationFunction(Enum):
    RELU = 1
    TANH = 2 # Hyperbolic tangent function
    SIGMOID = 3


def get_activation_function(activation):
    if activation == ActivationFunction.RELU:
        return ReLU()
    elif activation == ActivationFunction.TANH:
        return TanH()
    elif activation == ActivationFunction.SIGMOID:
        return Sigmoid()


#ReLU
class ReLU:
    def __init__(self):
        pass

    def eval(self,input):
        return np.maximum(0, input)

    def back_prop(self,input,d_chain,lam):
        d = input > 0
        return d_chain*d



class TanH:
    def __init__(self):
        pass

    def eval(self,input):
        return np.tanh(input)

    def back_prop(self,input, d_chain,lam):
        d = 1.0-np.tanh(input)*np.tanh(input)
        return d_chain * d



class Sigmoid:
    def __init__(self):
        pass

    def eval(self,input):
        return 1. / (1 + np.exp(-input))

    def back_prop(self,input, d_chain,lam):
        d = self.eval(input) * (1.0 - self.eval(input))
        return d_chain * d