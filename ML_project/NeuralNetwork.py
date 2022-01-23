import sklearn.metrics
import numpy
import math
import numpy as np
from MONK_data_processing import monk_output_processing
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")


class NeuralNetwork:
    """
    Class that represents a Neural Network
    """
    def __init__(self, layer_list, learning_rate, activation_function_hidden, activation_function_output, loss,
                 momentum, lambda_reg, reg_type="l2", rand=False, lr_type="fixed", tau=0):
        """
        Constructor of the Neural Network

        :param layer_list: topology of the Neural Network
        :param learning_rate: eta
        :param activation_function_hidden: activation function for the hidden layers
        :param activation_function_output: activation function for the output layer
        :param loss: loss function
        :param momentum: alpha, momentum coefficient
        :param lambda_reg: lambda, regularization coefficient
        :param reg_type: type of regularization, either "l1" or "l2", default "l2" for Tikhonov's regularization
        :param rand: boolean parameter, if true the NN is trained as a Randomized Neural Network, default False
        :param lr_type: specifies if learning rate is fixed or variable, default "fixed"
        :param tau: tau, value for variable learning rate
        """
        self.num_input = layer_list[0]
        self.learning_rate = learning_rate
        self.activation_function = activation_function_hidden
        self.output_activation_function = activation_function_output
        self.loss = loss
        self.momentum = momentum
        self.lambda_reg = lambda_reg
        self.num_inputs = layer_list[0]
        self.layer_arr = []  # Consists of hidden layers + output
        self.init_all_layers(layer_list)
        self.net_arr = [0 for x in range(len(layer_list) - 1)]
        self.lr_type = lr_type  # Possible types: "fixed", "linear_decay"
        self.tau = tau  # Limit step for learning rate in linear decay
        self.lr_base = learning_rate
        self.total_num_batch_trained = 0

        # Settings for regularization
        if lr_type == "linear_decay":
            self.lr_tau = 0.01 * self.lr_base

        if reg_type == "l2" or reg_type == "l1":
            self.reg_type = reg_type
        else:
            raise ValueError('Invalid Reg. Func.')

        # Settings for random
        self.rand = rand
        if not self.rand:
            self.old_grad = [0.0 for x in range(len(layer_list) - 1)]
            self.old_grad_bias = [0.0 for x in range(len(layer_list) - 1)]
        else:
            self.old_grad = [0.0]
            self.old_grad_bias = [0.0]

    def init_all_layers(self, layer_list):
        """
        Initializes all layers in the Neural Network

        :param layer_list: topology of the Neural Network
        """
        if len(layer_list) < 2:
            print("Error: not enough layers, must be at least 2!")
            return

        for i, num_neurons in enumerate(layer_list[1:]):  # Starts with the first hidden layer
            l = NeuronLayer(num_neurons, layer_list[i])
            self.layer_arr.append(l)
        return

    def upload_layer_weights(self, new_layer_weights_list, new_bias_list):
        """
        This function lets you upload specific values for the weights into the Neural Network

        :param new_layer_weights_list: new values for the weights
        :param new_bias_list: new values for the bias
        """
        for i, weights_arr in enumerate(new_layer_weights_list):
            self.layer_arr[i].upload_weights(weights_arr, new_bias_list[i])

    def __str__(self):
        """
        Returns information about the Neural Network in a string format
        """
        s = '['
        for i, l in enumerate(self.layer_arr):
            s += str(l.layer_weight_structure.shape[0]) + ', '
        s += str(self.layer_arr[-1].layer_weight_structure.shape[1]) + '] '
        s += ' | learning rate: ' + str(self.learning_rate)
        s += ' | act: ' + str(self.activation_function) + ' ' + str(self.output_activation_function)
        s += ' | loss: ' + str(self.loss)
        s += ' | alpha: ' + str(self.momentum)
        s += ' | lambda: ' + str(self.lambda_reg)
        s += ' | rand: ' + str(self.rand)
        return s

    def activation(self, x, out_act=False):
        """
        Computes the activation function given an input

        :param x: input
        :param out_act: boolean value, if True we must compute the output layer's activation function
                    otherwise the hidden layers' activation function, default False

        :return: computed value of the activation function applied to the input
        """
        # Selects either hidden layers' activation function or the output layer's activation function
        if out_act:
            act_fun = self.output_activation_function
        else:
            act_fun = self.activation_function

        if act_fun == "softmax":
            osum = np.exp(x).sum()
            return np.exp(x) / osum
        elif act_fun == "sigmoid":
            sig = np.divide(1, (1 + np.exp(-x)))
            return sig
        elif act_fun == "relu":
            return np.maximum(0, x)
        elif act_fun == "leakyrelu":
            for i in range(np.shape(x)[0]):
                x[i] = [j if j >= 0 else 0.01 * j for j in x[i]]
            return x
        elif act_fun == "tanh":
            return np.tanh(x)
        elif act_fun == "identity":
            return x

    def activation_derivative(self, x, out_act=False):
        """
        Computes the activation function's derivative given an input

        :param x: input
        :param out_act: boolean value, if True we must compute the output layer's activation function derivative
                    otherwise the hidden layers' activation function derivative, default False

        :return: computed value of the activation function derivative applied to the input
        """
        # Selects either hidden layers' activation function derivative or the output layer's activation function der.
        if out_act:
            act_fun = self.output_activation_function
        else:
            act_fun = self.activation_function

        if act_fun == "softmax":
            res = self.activation(x, True)
            return res * (1.0 - res)
        elif act_fun == "sigmoid":
            sig = self.activation(x)
            return numpy.multiply(sig, (1 - sig))
        elif act_fun == "relu":
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
        elif act_fun == "leakyrelu":
            for i in range(np.shape(x)[0]):
                x[i] = [1 if j > 0 else 0.01 for j in x[i]]
            return x
        elif act_fun == "tanh":
            return 1 - np.square(np.tanh(x))
        elif act_fun == "identity":
            return np.ones(np.shape(x))

    def regularization_derivative(self, lambd, weights):
        """
        This function computes the regularization derivative

        :param lambd: lambda, regularization coefficient
        :param weights: weights of the network
        """
        if self.reg_type == "l2":  # ridge
            return lambd * weights
        elif self.reg_type == "l1":  # lasso
            res = np.sign(weights) * lambd
            return res
        return

    def loss_function(self, h_output, y_target):
        """
        Computes the loss given a predicted output and the actual target values

        :param h_output: predicted output
        :param y_target: target values

        :return: computed loss function
        """
        if self.loss == "MEE":  # Mean Euclidean Error
            loss = (np.linalg.norm(np.subtract(h_output, y_target)))
            return loss
        elif self.loss == "MSE":  # Mean Square Error
            loss = np.sum((h_output - y_target) ** 2)
            return loss

    def feedforward(self, x_input):
        """
        This function performs a forward pass on the Neural Network

        :param x_input: input patterns

        :return: partial_result: the computed output, given the inputs
        """
        if len(np.shape(x_input)) == 1:
            partial_result = np.array([x_input], ndmin=2).T
        else:
            partial_result = x_input

        for i, layer in enumerate(self.layer_arr):
            net = np.matmul(layer.layer_weight_structure.T, partial_result)
            net = np.add(net, layer.bias)
            self.net_arr[i] = net
            partial_result = self.activation(net, i == (len(self.layer_arr) - 1))
        return partial_result

    def backpropagation(self, x_input, h_output, y_target):
        """
        This function performs a backward pass on the Neural Network

        :param x_input: input patterns
        :param h_output: predicted output
        :param y_target: target values

        :return: grad_k: computed gradient for the layer weights
        :return grad_k_bias: computed gradient for the layer biases
        """
        y_target = np.array([y_target]).T
        grad_k = [0 for x in range(len(self.layer_arr))]
        grad_k_bias = [0 for x in range(len(self.layer_arr))]
        # Computes output layer loss derivative
        dLoss = y_target - h_output

        for index_layer in reversed(range(len(self.layer_arr))):

            if index_layer != 0:  # if it's not the first layer
                output_u = self.activation(self.net_arr[index_layer - 1], index_layer == len(
                    self.layer_arr) - 1)  # it takes the previous layer's output
            else:
                output_u = np.array([x_input]).T  # otherwise it takes the input

            layer = self.layer_arr[index_layer]

            # Computes the derivative of the net
            dNet = self.activation_derivative(self.net_arr[index_layer], index_layer == len(self.layer_arr) - 1)
            delta_k = np.multiply(dLoss, dNet)

            # Computes the delta of the weights
            Dw = np.matmul(delta_k, output_u.T)

            # Computes the gradient
            grad_k[index_layer] = self.learning_rate * Dw.T
            grad_k_bias[index_layer] = np.sum(delta_k, axis=1, keepdims=True)

            if self.rand:
                break  # performs only backpropagation on the last layer then exists for loop

            # If we still have layers to cycle through
            if index_layer != 0:
                dLoss = []
                for i in range(layer.layer_weight_structure.shape[0]):
                    # Compute the derivative of the loss for the hidden layers
                    dLoss.append(np.dot(delta_k.T, layer.layer_weight_structure[i].reshape(
                        (layer.layer_weight_structure[i].shape[0], 1))))
                dLoss = np.hstack((dLoss)).T  # Reshapes the matrix correctly

        return grad_k, grad_k_bias

    def stopping_criteria(self, curr_epoch, loss_for_epochs, val_loss_arr, val_input, val_target):
        """
        Function that determines if the stopping criteria is met or not, both are evaluated over a certain number
        of consecutive epochs

        :param curr_epoch: number of current epoch
        :param loss_for_epochs: training loss over the epochs
        :param val_loss_arr: validation loss over the epochs
        :param val_input: validation set input values
        :param val_target: validation set target values

        :return: True if stopping criteria is met, otherwise False
        """
        if curr_epoch > 2:
            # Stopping based on TR loss decay, when it becomes asymptotic/there are no major shifts it stops
            diff = np.abs((loss_for_epochs[-1] - loss_for_epochs[-2]))

            # If there are no major shifts in TR loss
            if diff < 1e-5 and len(val_input) <= 0:
                self.count_decaying_loss += 1  # increment counter
            else:
                self.count_decaying_loss = 0  # reset counter
            # If we have reached the limit
            if self.count_decaying_loss >= self.limit_decaying_loss and len(val_input) <= 0:
                return True

        # Early stopping based on validation set loss, checked every 5 epochs
        if curr_epoch % 5 == 0 and curr_epoch != 0 and len(val_input) > 0 and len(val_target) > 0:
            # Compute validation loss
            val_output, val_loss = self.predict(val_input, val_target)
            val_loss_arr.append(val_loss)

            # If loss is increasing
            if self.val_loss_prev < val_loss:
                self.count_validation_loss += 1  # increment counter

                # If we have reached the limit
                if self.count_validation_loss >= self.limit_validation_loss:
                    # Reset to previous weights
                    self.layer_arr = [deepcopy(x) for x in self.previous_weights]
                    return True
            else:
                self.val_loss_prev = val_loss
                self.count_validation_loss = 0  # reset counter
                if self.count_validation_loss == 0:
                    # Save weights now that validation loss is not increasing
                    self.previous_weights = deepcopy(self.layer_arr)

            return False

    def train(self, x_input, y_target, num_epochs=300, batch_size=1, val_input=[], val_target=[], test_input=[],
              test_target=[], accuracy_flag=False):
        """
        This function trains the Neural Network, computing loss and accuracy (if flag is True) for each epoch and the
        predicted outputs

        :param x_input: input patterns
        :param y_target: target values
        :param num_epochs: maximum number of epochs, default 300
        :param batch_size: number of patterns for each batch, default 1 (online)
        :param val_input: Validation Set input, default empty
        :param val_target: Validation set target values, default empty
        :param test_input: Test set input, default empty
        :param test_target: Test set target values, default empty
        :param accuracy_flag: if True the accuracy is computed, default False

        :return: h_output: predicted output values
        :return: loss_for_epochs: loss values, one for each epoch
        :return: accuracy_for_epochs: accuracy values, one for each epoch
        :return: val_loss_every_5_epochs: loss on Validation set, one every 5 epochs
        :return: test_loss_for_epochs: loss on Test set, one for each epoch
        :return: test_accuracy_for_epochs: accuracy on Test set, one for each epoch
        """
        x_input = np.array(x_input)
        y_target = np.array(y_target)
        i = 0
        num_iterations = 0

        self.count_decaying_loss = 0
        self.limit_decaying_loss = num_epochs * 0.05  # 5% of the total number of epochs

        self.count_validation_loss = 0
        self.limit_validation_loss = num_epochs * 0.05  # 5% of the total number of epochs

        self.val_loss_prev = math.inf

        loss_for_epochs = []
        val_loss_every_5_epochs = []
        test_loss_for_epochs = []

        accuracy_for_epochs = []
        test_accuracy_for_epochs = []

        num_patterns = x_input.shape[1]
        num_batch = np.ceil(x_input.shape[1] / batch_size)

        try:

            # Stops after a certain number of epochs, to avoid diverging
            while i < num_epochs:
                if i % 100 == 0:
                    print("Number of current epoch: ", i)

                # Checks if a stopping criteria is met, in which case it exits
                if self.stopping_criteria(i, loss_for_epochs, val_loss_every_5_epochs, val_input, val_target):
                    break

                # Split into mini-batches
                input_batches = np.array_split(x_input, num_batch, axis=1)
                target_batches = np.array_split(y_target, num_batch, axis=1)

                for batch in range(len(input_batches)):

                    grad_net = [0 for x in range(len(self.layer_arr))]
                    grad_net_bias = [0 for x in range(len(self.layer_arr))]

                    # Initializes the gradient matrices
                    for j, l in enumerate(self.layer_arr):
                        grad_net[j] = np.zeros(l.layer_weight_structure.shape)
                        grad_net_bias[j] = np.zeros(l.bias.shape)

                    for p in range(np.shape(input_batches[batch])[1]):
                        h_output = self.feedforward(input_batches[batch][:, p])

                        grad, grad_bias = self.backpropagation(input_batches[batch][:, p], h_output,
                                                               target_batches[batch][:, p])

                        # Compute gradient
                        grad_net = np.add(grad_net, grad)
                        grad_net_bias = np.add(grad_net_bias, grad_bias)

                    if batch_size != 1:
                        # Compute gradients
                        if batch_size == num_patterns:
                            grad_net = grad_net / num_patterns
                            grad_net_bias = grad_net_bias / num_patterns
                        else:
                            grad_net = grad_net / np.shape(input_batches[batch])[1]
                            grad_net_bias = grad_net_bias / np.shape(input_batches[batch])[1]

                    # Update learning rate
                    self.update_learning_rate(num_iterations)
                    # Update weights of the Neural Network
                    self.update_weights(grad_net, grad_net_bias, batch_size, num_patterns)
                    num_iterations += 1  # Step increment

                # Computes loss for the epoch
                h_output = self.feedforward(x_input)
                l = self.loss_function(h_output, y_target)
                loss_for_epochs.append(l / np.shape(x_input)[1])

                if np.isnan(h_output).any():
                    raise ValueError("Overflow")

                # Computes accuracy
                if accuracy_flag:
                    p_output = monk_output_processing(h_output[0])
                    accuracy_for_epochs.append(sklearn.metrics.accuracy_score(y_target[0], p_output))

                # Compute Test prediction and error
                if len(test_input) > 0:
                    t_output, t_loss = self.predict(test_input, test_target)
                    test_loss_for_epochs.append(t_loss)
                    if accuracy_flag:
                        t_output = monk_output_processing(t_output[0])
                        test_accuracy_for_epochs.append(sklearn.metrics.accuracy_score(test_target[0], t_output))

                i += 1  # Epoch Increment

        except ValueError as ve:
            print(ve)
            print(str(self))
            return [], [], [], [], [], []

        return h_output, loss_for_epochs, accuracy_for_epochs, val_loss_every_5_epochs, \
               test_loss_for_epochs, test_accuracy_for_epochs

    def predict(self, x_input, y_target=[]):
        """
        Function that predicts outputs given some inputs, if targets are passed then the error is computed

        :param x_input: input values
        :param y_target: target values, default [] empty list

        :return: h_output: predicted output values
        :return: loss: computed error or empty list if no target was passed as input parameter
        """
        loss = []
        h_output = self.feedforward(x_input)
        if len(y_target) > 0:
            loss = self.loss_function(h_output, y_target)
            loss /= np.shape(x_input)[1]
        return h_output, loss

    def update_learning_rate(self, curr_step):
        """
        Function that updates the learning rate

        :param curr_step: current step
        """
        if self.lr_type == "linear_decay":
            self.linear_lr_decay(curr_step)
        else:
            return

    def linear_lr_decay(self, curr_step):
        """
        Computes the linear decay of the learning rate

        :param curr_step: current step
        """
        if curr_step < self.tau and self.learning_rate > self.lr_tau:
            alpha_step = curr_step / self.tau
            self.learning_rate = (1. - alpha_step) * self.lr_base + alpha_step * self.lr_tau
        return

    def update_weights(self, grad_weights, grad_bias, batch_size, num_patterns):
        """
        This function updates the weights of the Neural Network's layers

        :param grad_weights: gradient of the weights of the layers
        :param grad_bias: gradient of the bias of the layers
        :param batch_size: number of patterns for each batch
        :param num_patterns: number of total patterns
        """
        for i, layer in enumerate(self.layer_arr):

            if self.rand: # If random is enabled, only update the output layer weights
                i = -1
                layer = self.layer_arr[i]

            # Computes update taking into account momentum...
            grad_and_momentum = grad_weights[i] + self.momentum * self.old_grad[i]
            self.old_grad[i] = grad_and_momentum
            # ... and regularization
            layer.layer_weight_structure += (grad_and_momentum -
                                             self.regularization_derivative(self.lambda_reg * (batch_size / num_patterns),
                                                                        layer.layer_weight_structure))

            grad_and_momentum_bias = grad_bias[i] + self.momentum * self.old_grad_bias[i]

            self.old_grad_bias[i] = grad_and_momentum_bias

            layer.bias += grad_and_momentum_bias

            if self.rand:
                break

        return


class NeuronLayer:
    """
    Class representing a Neural Network's layer
    """
    def __init__(self, num_neurons, num_weights):
        """
        Constructor for the NeuronLayer object, sets up the weight structure with randomized weights

        :param num_neurons: number of neurons for the layer
        :param num_weights: number of weights for each neuron
        """
        self.num_neurons = num_neurons
        self.num_weights = num_weights
        max_val, min_val = 0.7, -0.7  # Initialization range
        # Every column of the matrix has the weights of each neuron in the layer and every row is the number of neurons
        self.layer_weight_structure = np.random.uniform(min_val, max_val,
                                                        (num_weights, num_neurons))

        # Bias for the layer
        self.bias = \
            np.random.uniform(min_val, max_val, (num_neurons, 1))

    def upload_weights(self, weights_arr, bias_arr):
        """
        Upload a set of weights for a layer instead of the randomic ones.
        Make check on weight array dimensions.
        Used for initializing a NN with already trained weights.

        :param weights_arr: array of uploaded weights for each neuron in the layer
        :param bias_arr: bias to load
        """
        if np.shape(self.layer_weight_structure) == np.shape(weights_arr):
            self.layer_weight_structure = weights_arr
            self.bias = bias_arr
        else:
            print("Shape of weights to insert is not valid,",
                  np.shape(weights_arr), " instead of", np.shape(self.layer_weight_structure))

    def __str__(self):
        """
        Prints the shape of the layer and the weights
        """
        return "shape: " + str(np.shape(self.layer_weight_structure)) + " " + str(self.layer_weight_structure)


np.set_printoptions(threshold=np.inf)
