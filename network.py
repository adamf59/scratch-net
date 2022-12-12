import numpy as np
import time

class NeuralNetwork:

    def __init__(self) -> None:
        
        self.b_mat = [ ]
        """List containing the bias matrix b_mat[i] for the layer i, type `np.ndarray`"""
        
        self.w_mat = [ ]
        """List containing the edge weights w_mat[i] leading out of layer i, type `np.ndarray`"""
        
        self.n_mat = [ ]
        """List containing the neuron/node activations n_mat[i] for the layer i, type `np.ndarray`"""

        self.alpha_mat = [ ]
        """List containing the edge activations of layer i, type `np.ndarray`"""

        self.structure = [ ]


    def compute_n_mat(self, layer_i: int) -> np.ndarray:
        """
        Recomputes the node activation matrix for a given layer.

        layer_i: which layer to find activations for [ >=1 ] (not defined for the input layer)
        """
        # layer check
        if layer_i < 1:
            raise RuntimeError("attempted to compute n_mat on invalid layer.")
        # get the alpha matrix from the previous layer
        alpha: np.ndarray = self.alpha_mat[layer_i - 1]
        # get the b matrix for this layer
        b: np.ndarray = self.b_mat[layer_i]
        # find the sum
        n: np.ndarray = np.add(alpha, b)
        # apply activation function
        activation_function: function = self.structure[layer_i][1]
        n = activation_function(n)
        # update state
        self.n_mat[layer_i] = n
        # return state
        return self.n_mat[layer_i]

    def compute_alpha_mat(self, layer_i: int) -> np.ndarray:
        """
        Recomputes the edge activation matrix (alpha) for a given layer.

        layer_i: which layer to find edge activations for [ >=0 ]
        """
        # get the weight matrix for layer i
        w: np.ndarray = self.w_mat[layer_i]
        # get the source activation
        n: np.ndarray = self.n_mat[layer_i]
        # compute the alpha
        alpha: np.ndarray = np.dot(n, w)
        # update state
        self.alpha_mat[layer_i] = alpha
        # return state
        return self.alpha_mat[layer_i]


    def evaluate(self, input: np.ndarray) -> np.ndarray:
        """
        Evaluates the input, which should be of the same size as the layer 0, and returns the output activations of the network `n_mat[-1]`.
        """

        if len(input) != self.structure[0][0]:
            raise RuntimeError("Cannot evaluate because input sizes do not match.")
        
        # Set the n_mat for the first layer to the input array
        self.n_mat[0] = input
        # for each layer, compute the alpha and n_mat
        for l in range(len(self.structure) - 1):
            
            # compute the alpha
            self.compute_alpha_mat(l)
            # compute the n_mat for the next layer
            self.compute_n_mat(l + 1)

        # return the n_mat for the last layer
        return self.n_mat[-1]

    def evaluate_timed(self, input: np.ndarray) -> np.ndarray:
        
        _s = time.time()
        output = self.evaluate(input)
        _e = time.time()

        print("evaluated network in", (_e - _s), "sec")

        return output

    def train_batch(self, data_batch: list) -> float:
        """
        Trains the neural network using backpropagation, and returns the average of the loss function applied across all training data.

        data: list of tuples like ((`np.ndarray`) input, (`int`) class_label)
        """
        train_n = len(data_batch)
        
        loss_scores = np.zeros((1, train_n)) # the loss of each training item, of length 
        
    def evaluate_with_loss(self, input: np.ndarray, class_label: int) -> tuple:
        """
        Computes the output of the network, as well as the loss (or cost) associated with the input.
        See `evaluate` for `input` argument.

        Returns a tuple like (`np.ndarray`, `float`), where index 0 is the network output, and index 1 is the loss.
        """
        # compute the loss of the network using the Mean Square Error (MSE)
        observed: np.ndarray = self.evaluate(input) # evaluate the network's output
        predicted: np.ndarray = np.zeros(observed.shape) # set all labels to 0
        predicted[class_label] = 1.00 # set the correct label prediction to 1
        
        loss = (observed-predicted) ** 2

        return (observed, loss)    

class FCNeuralNetwork(NeuralNetwork):

    def __init__(self, structure: list) -> None:
        """
        Create a new perceptron network with full connectivity.

        structure: a list of tuples of the form (numNodes, actFunc)
        i.e. for a network with 100 inputs, 10 outputs, and 2 hidden layers
        with 6 nodes each, do:
            `[(100, None), (6, SigmoidFunction), (6, SigmoidFunction), (10, ReLU)]`
        """
        # super class init
        NeuralNetwork.__init__(self)
        # update number of layers
        self.structure = structure
        
        # unpack the structure
        for i, layer in enumerate(structure):
            # get the node count and activation function
            nodec, activation_function = layer
            # create n_mat, b_mat, and w_mat for each layer
            self.n_mat.append(np.full((1, nodec), 0))
            self.b_mat.append(np.full((1, nodec), 0))
            if i < len(structure) - 1:
                self.w_mat.append(np.random.uniform(-1, 1,(nodec, structure[i + 1][0])))
                self.alpha_mat.append(None)