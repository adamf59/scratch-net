# imports
from network import FCNeuralNetwork
from activations import SigmoidFunction
import numpy as np

# create a new network object

mynet = [
    (100, None),
    (16, SigmoidFunction),
    (16, SigmoidFunction),
    (10, SigmoidFunction)
]

network = FCNeuralNetwork(mynet)

input = np.array([
    0.50, 0.61, 0.05, 0.82, 0.73, 0.99, 0.19, 0.83, 0.23, 0.62,
    0.55, 0.28, 0.79, 0.13, 0.15, 0.93, 0.52, 0.13, 0.89, 0.98,
    0.33, 0.48, 0.67, 0.27, 0.07, 0.41, 0.47, 0.85, 0.59, 0.08,
    0.67, 0.83, 0.41, 0.64, 0.59, 0.57, 0.17, 0.24, 0.20, 0.28,
    0.42, 0.37, 0.57, 0.49, 0.61, 0.11, 0.90, 0.33, 0.73, 0.19,
    0.94, 0.86, 0.68, 0.40, 0.37, 0.12, 0.72, 0.55, 0.59, 0.21,
    0.11, 0.19, 0.38, 0.27, 0.76, 0.00, 0.57, 0.84, 0.92, 0.99,
    0.48, 0.84, 0.07, 0.32, 0.01, 0.80, 0.03, 0.02, 0.03, 0.68,
    0.48, 0.26, 0.79, 0.68, 0.08, 0.71, 0.93, 0.44, 0.15, 0.70,
    0.53, 0.49, 0.57, 0.93, 0.99, 0.05, 0.33, 0.69, 0.13, 0.78
    ])

print(network.evaluate_timed(input))


