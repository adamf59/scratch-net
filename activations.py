import numpy as np
import math

def _sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
SigmoidFunction = np.vectorize(_sigmoid)
