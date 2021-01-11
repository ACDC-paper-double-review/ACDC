from MyUtil import MyUtil as MyUtil
from ElasticNodes import ElasticNodes
from MySingletons import MyDevice

import numpy as np
import torch


# class ReverseLayerFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(self, x, alpha=1.0):
#         self.alpha = alpha
#
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(self, grad_output):
#         output = grad_output.neg() * self.alpha
#
#         return output, None


class NeuralNetwork(ElasticNodes):
    layers = None
    layer_value = None
    output_layer_value = None

    weight = None
    bias = None
    momentum = None
    bias_momentum = None

    output_weight = None
    output_bias = None
    output_momentum = None
    output_bias_momentum = None

    activation_function = None
    output_activation_function = None
    loss_function = None

    learning_rate = 0.01
    momentum_rate = 0.95

    error_value = None
    loss_value = None
    classification_rate = None
    misclassified = None

    output_beta = None
    output_beta_decreasing_factor = None

    __Eh = None
    __Eh2 = None

    @property
    def number_hidden_layers(self):
        return len(self.layers) - 2

    @property
    def input_size(self):
        return self.layers[0]

    @property
    def output_size(self):
        return self.layers[-1]

    @property
    def output(self):
        return self.output_layer_value

    @property
    def raw_output(self):
        return torch.max(self.output, axis=1)

    @property
    def outputed_classes(self):
        return torch.argmax(self.output, axis=1)

    @property
    def residual_error(self):
        return 1 - self.raw_output.values

    ACTIVATION_FUNCTION_AFFINE = 1
    ACTIVATION_FUNCTION_SIGMOID = ACTIVATION_FUNCTION_AFFINE + 1
    ACTIVATION_FUNCTION_TANH = ACTIVATION_FUNCTION_SIGMOID + 1
    ACTIVATION_FUNCTION_RELU = ACTIVATION_FUNCTION_TANH + 1
    ACTIVATION_FUNCTION_LINEAR = ACTIVATION_FUNCTION_RELU + 1
    ACTIVATION_FUNCTION_SOFTMAX = ACTIVATION_FUNCTION_LINEAR + 1
    ACTIVATION_FUNCTION_REVERSE_LAYER = ACTIVATION_FUNCTION_SOFTMAX + 1

    LOSS_FUNCTION_MSE = ACTIVATION_FUNCTION_REVERSE_LAYER + 1
    LOSS_FUNCTION_CROSS_ENTROPY = LOSS_FUNCTION_MSE + 1

    PRUNE_NODE_STRATEGY_SINGLE = LOSS_FUNCTION_CROSS_ENTROPY + 1
    PRUNE_NODE_STRATEGY_MULTIPLE = PRUNE_NODE_STRATEGY_SINGLE + 1

    def __init__(self, layers: list, init_weights: bool = True):
        self.layers = layers

        self.weight = []
        self.bias = []
        self.momentum = []
        self.bias_momentum = []
        self.activation_function = []

        for i in range(self.number_hidden_layers):
            nodes_before = layers[i]
            nodes_after = layers[i + 1]

            if init_weights:
                self.weight.append(self.xavier_weight_initialization(nodes_after, nodes_before))
                self.bias.append(self.xavier_weight_initialization(1, nodes_after))
                self.momentum.append(torch.zeros(self.weight[i].shape, dtype=torch.float, device=MyDevice().get()))
                self.bias_momentum.append(torch.zeros(self.bias[i].shape, dtype=torch.float, device=MyDevice().get()))
            else:
                self.weight.append(None)
                self.bias.append(None)
                self.momentum.append(None)
                self.bias_momentum.append(None)
                self.momentum_rate = 0

            self.activation_function.append(self.ACTIVATION_FUNCTION_SIGMOID)

        if init_weights:
            nodes_before = layers[-2]
            nodes_after = layers[-1]

            self.output_weight = self.xavier_weight_initialization(nodes_after, nodes_before)
            self.output_bias = self.xavier_weight_initialization(1, nodes_after)
            self.output_momentum = torch.zeros(self.output_weight.shape, dtype=torch.float, device=MyDevice().get())
            self.output_bias_momentum = torch.zeros(self.output_bias.shape, dtype=torch.float, device=MyDevice().get())
        else:
            self.output_weight = None
            self.output_bias = None
            self.output_momentum = None
            self.output_bias_momentum = None
            self.momentum_rate = 0

        self.output_activation_function = self.ACTIVATION_FUNCTION_SOFTMAX
        self.loss_function = self.LOSS_FUNCTION_CROSS_ENTROPY

        ElasticNodes.__init__(self, len(self.layers))

    ##### Weight initializations #####

    def xavier_weight_initialization(self, n_out: int, n_in: int, uniform: bool = False):
        if uniform:
            return torch.nn.init.xavier_uniform(tensor=torch.zeros(int(n_out), int(n_in), dtype=torch.float,
                                                                   requires_grad=True, device=MyDevice().get()))
        return torch.nn.init.xavier_normal_(tensor=torch.zeros(int(n_out), int(n_in), dtype=torch.float,
                                                               requires_grad=True, device=MyDevice().get()))

    def he_weight_initialization(self, n_out, n_in, shape=None):
        #TODO
        mean = 0.0
        sigma = np.sqrt(2 / n_in)
        if shape is None:
            shape = (n_out, n_in)
        return np.random.normal(mean, sigma, shape)

    ##### Noise #####

    def masking_noise(self, x: torch.tensor, noise_ratio: float = 0.0):
        return x.detach().masked_fill(torch.rand(x.shape, device=MyDevice().get()) <= noise_ratio, 0)

    ##### Activation functions #####

    @staticmethod
    def sigmoid(z: torch.tensor):
        return torch.sigmoid(z)

    @staticmethod
    def tanh(z):
        return torch.tanh(z)

    @staticmethod
    def relu(z):
        return torch.nn.functional.relu(z)

    @staticmethod
    def linear(layer_value: torch.tensor, weight: torch.tensor, bias: torch.tensor):
        return torch.nn.functional.linear(layer_value, weight, bias)

    @staticmethod
    def softmax(z, axis: int = 1):
        return torch.nn.functional.softmax(z, dim=axis)

    def reset_grad(self):
        for i in range(self.number_hidden_layers):
            self.weight[i] = self.weight[i].detach()
            self.bias[i] = self.bias[i].detach()
            self.weight[i].requires_grad = True
            self.bias[i].requires_grad = True

        self.output_weight = self.output_weight.detach()
        self.output_bias = self.output_bias.detach()
        self.output_weight.requires_grad = True
        self.output_bias.requires_grad = True

    def feedforward(self, x: torch.Tensor, y: torch.Tensor, train: bool = False):
        return self.forward_pass(x, train=train).calculate_error(y)

    def backpropagate(self):
        self.loss_value.backward()

        return self

    def test(self, x: torch.Tensor, y: torch.Tensor, is_beta_updatable: bool = False):
        self.feedforward(x=x, y=y)

        m = y.shape[0]

        true_classes = torch.argmax(y, axis=1)
        self.misclassified = torch.sum(torch.ne(self.outputed_classes, true_classes)).item()
        self.classification_rate = 1 - self.misclassified / m

        if is_beta_updatable:
            class_label = self.output_layer_value.max(axis=2)
            for i in range(m):
                if self.true_classes[i] == class_label[i]:
                    self.output_beta = np.max(self.output_beta * self.output_beta_decreasing_factor, 0)
                    self.output_beta_decreasing_factor = np.max(self.output_beta_decreasing_factor - 0.01, 0)
                else:
                    self.output_beta = max(self.output_beta * (1 + self.output_beta_decreasing_factor), 1)
                    self.output_beta_decreasing_factor = max(self.output_beta_decreasing_factor + 0.01, 1)

        return self

    def train(self, x: torch.Tensor, y: torch.Tensor, weight_no: int = None, is_neg_grad: bool = False):
        self.feedforward(x=x, y=y, train=True).backpropagate()

        if weight_no is None:
            for weight_no in range(self.number_hidden_layers, -1, -1):
                self.update_weight(weight_no=weight_no, is_neg_grad=is_neg_grad)
        else:
            self.update_weight(weight_no=weight_no, is_neg_grad=is_neg_grad)

    def update_weight(self, weight_no: int, is_neg_grad: bool = False):
        if weight_no >= self.number_hidden_layers:
            dW: torch.Tensor = self.learning_rate * self.output_weight.grad
            db: torch.Tensor = self.learning_rate * self.output_bias.grad
            if self.momentum_rate > 0:
                self.output_momentum: torch.Tensor = self.momentum_rate * self.output_momentum + dW
                self.output_bias_momentum: torch.Tensor = self.momentum_rate * self.output_bias_momentum + db
                dW: torch.Tensor = self.output_momentum
                db: torch.Tensor = self.output_bias_momentum
            if is_neg_grad:
                self.output_weight: torch.Tensor = self.output_weight - dW.neg()
                self.output_bias: torch.Tensor = self.output_bias - db.neg()
            else:
                self.output_weight: torch.Tensor = self.output_weight - dW
                self.output_bias: torch.Tensor = self.output_bias - db
        else:
            dW: torch.Tensor = self.learning_rate * self.weight[weight_no].grad
            db: torch.Tensor = self.learning_rate * self.bias[weight_no].grad
            if self.momentum_rate > 0:
                self.momentum[weight_no]: torch.Tensor = self.momentum_rate * self.momentum[weight_no] + dW
                self.bias_momentum[weight_no]: torch.Tensor = self.momentum_rate * self.bias_momentum[weight_no] + db
                dW: torch.Tensor = self.momentum[weight_no]
                db: torch.Tensor = self.bias_momentum[weight_no]
            if is_neg_grad:
                self.weight[weight_no]: torch.Tensor = self.weight[weight_no] - dW.neg()
                self.bias[weight_no]: torch.Tensor = self.bias[weight_no] - db.neg()
            else:
                self.weight[weight_no]: torch.Tensor = self.weight[weight_no] - dW
                self.bias[weight_no]: torch.Tensor = self.bias[weight_no] - db

    def forward_pass(self, x: torch.Tensor, train: bool = False):
        if train:
            self.reset_grad()
        self.layer_value = []
        self.layer_value.append(x)

        for i in range(self.number_hidden_layers):
            if self.activation_function[i] == self.ACTIVATION_FUNCTION_AFFINE:
                self.layer_value.append(self.linear(self.layer_value[i], self.weight[i], self.bias[i]))
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_SIGMOID:
                self.layer_value.append(self.sigmoid(self.linear(self.layer_value[i], self.weight[i], self.bias[i])))
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_TANH:
                self.layer_value.append(self.tanh(self.linear(self.layer_value[i], self.weight[i], self.bias[i])))
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_RELU:
                self.layer_value.append(self.relu(self.linear(self.layer_value[i], self.weight[i], self.bias[i])))
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_LINEAR:
                raise TypeError('Not implemented')
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_SOFTMAX:
                self.layer_value.append(self.softmax(self.linear(self.layer_value[i], self.weight[i], self.bias[i])))
            elif self.activation_function[i] == self.ACTIVATION_FUNCTION_REVERSE_LAYER:
                self.layer_value.append(self.reverse_layer(self.layer_value[i]))

        if self.output_activation_function == self.ACTIVATION_FUNCTION_AFFINE:
            self.output_layer_value = self.linear(self.layer_value[-1], self.output_weight, self.output_bias)
        elif self.output_activation_function == self.ACTIVATION_FUNCTION_SIGMOID:
            self.output_layer_value = self.sigmoid(self.linear(self.layer_value[-1], self.output_weight, self.output_bias))
        elif self.output_activation_function == self.ACTIVATION_FUNCTION_TANH:
            self.output_layer_value = self.tanh(self.linear(self.layer_value[-1], self.output_weight, self.output_bias))
        elif self.output_activation_function == self.ACTIVATION_FUNCTION_RELU:
            self.output_layer_value = self.relu(self.linear(self.layer_value[-1], self.output_weight, self.output_bias))
        elif self.output_activation_function == self.ACTIVATION_FUNCTION_SOFTMAX:
            self.output_layer_value = self.softmax(self.linear(self.layer_value[-1], self.output_weight, self.output_bias), axis=1)
        elif self.output_activation_function == self.ACTIVATION_FUNCTION_REVERSE_LAYER:
            self.output_layer_value = self.reverse_layer(self.layer_value[-1])

        return self

    def calculate_error(self, y: torch.tensor):
        self.error_value = y - self.output_layer_value

        if self.loss_function == self.LOSS_FUNCTION_MSE:
            self.loss_value = torch.nn.functional.mse_loss(self.output_layer_value, y)
        elif self.loss_function == self.LOSS_FUNCTION_CROSS_ENTROPY:
            self.loss_value = torch.nn.functional.cross_entropy(self.output_layer_value, torch.argmax(y, 1))

        return self

    def compute_expected_values(self, in_place: bool = False):
        self.data_mean, self.data_variance, self.data_standard_deviation = \
            MyUtil.recursive_mean_standard_deviation(self.layer_value[0],
                                                            self.data_mean,
                                                            self.data_variance,
                                                            self.number_samples_feed)

        self.Eh, self.Eh2 = self.compute_inbound_expected_values()

    def compute_inbound_expected_values(self, number_hidden_layer: int = None):
        nhl = number_hidden_layer  # readability
        if nhl is None:
            nhl = self.number_hidden_layers - 1

        if nhl == 0:
            inference, center, std = (1, self.data_mean, self.data_standard_deviation)
            py = MyUtil.probit(center, std)
            Eh = inference * self.sigmoid(self.linear(self.weight[0], py, self.bias[0].T))
        else:
            Eh, _ = self.compute_inbound_expected_values(number_hidden_layer=nhl - 1)
            weight, bias = (self.weight[nhl], self.bias[nhl]) if nhl < self.number_hidden_layers + 1 else (self.output_weight, self.output_bias)
            Eh = self.sigmoid(self.linear(weight, Eh.T, bias.T))

        return Eh, Eh ** 2

    @property
    def Eh(self):
        return self.__Eh

    @Eh.setter
    def Eh(self, value: torch.tensor):
        self.__Eh = value

    @property
    def Eh2(self):
        return self.__Eh2

    @Eh2.setter
    def Eh2(self, value: torch.tensor):
        self.__Eh2 = value

    @property
    def Ey(self):
        return self.softmax(self.linear(self.output_weight, self.Eh.T, self.output_bias.T), axis=0)

    @property
    def Ey2(self):
        return self.softmax(self.linear(self.output_weight, self.Eh2.T, self.output_bias.T), axis=0)

    @property
    def network_variance(self):
        return MyUtil.frobenius_norm(self.Ey2 - self.Ey ** 2)

    def compute_bias(self, y):
        return MyUtil.frobenius_norm((self.Ey.T - y) ** 2)

    def width_adaptation_stepwise(self, y, prune_strategy: int = None):
        if prune_strategy is None:
            prune_strategy = self.PRUNE_NODE_STRATEGY_MULTIPLE

        nhl: int = self.number_hidden_layers

        self.number_samples_feed = self.number_samples_feed + 1
        self.number_samples_layer[nhl] = self.number_samples_layer[nhl] + 1
        self.compute_expected_values()

        self.bias_mean[nhl], self.bias_variance[nhl], self.bias_standard_deviation[nhl] = \
            MyUtil.recursive_mean_standard_deviation(self.compute_bias(y),
                                                     self.bias_mean[nhl],
                                                     self.bias_variance[nhl],
                                                     self.number_samples_feed)

        self.var_mean[nhl], self.var_variance[nhl], self.var_standard_deviation[nhl] = \
            MyUtil.recursive_mean_standard_deviation(self.network_variance,
                                                     self.var_mean[nhl],
                                                     self.var_variance[nhl],
                                                     self.number_samples_feed)

        if self.number_samples_layer[nhl] <= 1 or self.growable[nhl]:
            self.minimum_bias_mean[nhl] = self.bias_mean[nhl]
            self.minimum_bias_standard_deviation[nhl] = self.bias_standard_deviation[nhl]
        else:
            self.minimum_bias_mean[nhl] = np.min([self.minimum_bias_mean[nhl], self.bias_mean[nhl]])
            self.minimum_bias_standard_deviation[nhl] = np.min([self.minimum_bias_standard_deviation[nhl], self.bias_standard_deviation[nhl]])

        if self.number_samples_layer[nhl] <= self.input_size + 1 or self.prunable[nhl][0] != -1:
            self.minimum_var_mean[nhl] = self.var_mean[nhl]
            self.minimum_var_standard_deviation[nhl] = self.var_standard_deviation[nhl]
        else:
            self.minimum_var_mean[nhl] = np.min([self.minimum_var_mean[nhl], self.var_mean[nhl]])
            self.minimum_var_standard_deviation[nhl] = np.min([self.minimum_var_standard_deviation[nhl], self.var_standard_deviation[nhl]])

        self.BIAS.append(self.bias_mean[nhl])
        self.VAR.append(self.var_mean[nhl])

        if self.output_size == 512:  # STL or CIFAR
            alpha_1 = 1.45
            alpha_2 = 0.95
        else:
            alpha_1 = 1.25
            alpha_2 = 0.75

        self.growable[nhl] = self.is_growable(self.compute_bias(y), alpha_1, alpha_2)
        self.prunable[nhl] = self.is_prunable(prune_strategy, 2 * alpha_1, 2 * alpha_2)

    def is_growable(self, bias: torch.tensor, alpha_1: float = 1.25, alpha_2: float = 0.75):
        nhl = self.number_hidden_layers  # readability

        current = self.bias_mean[nhl] + self.bias_standard_deviation[nhl]
        biased_min = self.minimum_bias_mean[nhl] \
                     + (alpha_1 * torch.exp(-bias) + alpha_2) * self.minimum_bias_standard_deviation[nhl]

        if self.number_samples_layer[nhl] > 1 and current >= biased_min:
            return True
        return False

    def is_prunable(self, prune_strategy: int = None, alpha_1: float = 2.5, alpha_2: float = 1.5):
        if prune_strategy is None:
            prune_strategy = self.PRUNE_NODE_STRATEGY_MULTIPLE
        nhl = self.number_hidden_layers  # readability

        current = self.var_mean[nhl] + self.var_standard_deviation[nhl]
        biased_min = self.minimum_var_mean[nhl] \
                     + (alpha_1 * torch.exp(-self.network_variance) + alpha_2) * self.minimum_var_standard_deviation[nhl]

        if not self.growable[nhl] \
                and self.layers[nhl] > 1 \
                and self.number_samples_layer[nhl] > self.input_size + 1 \
                and current >= biased_min:

            if prune_strategy == self.PRUNE_NODE_STRATEGY_SINGLE:
                return torch.argmin(self.Eh)
            elif prune_strategy == self.PRUNE_NODE_STRATEGY_MULTIPLE:
                nodes_to_prune = torch.where(self.Eh < torch.abs(torch.mean(self.Eh) - torch.var(self.Eh)))
                if len(nodes_to_prune[0]):
                    return nodes_to_prune[0]
                else:
                    return torch.argmin(self.Eh)

        return [-1]

    def grow_node(self, layer_number: int):
        self.layers[layer_number] += 1
        if layer_number >= 0:
            self.grow_weight_row(layer_number - 1)
            self.grow_bias(layer_number - 1)
        if layer_number <= self.number_hidden_layers:
            self.grow_weight_column(layer_number)

    def grow_weight_row(self, layer_number: int):
        def add_element(tensor_data: torch.tensor, momentum_tensor_data: torch.tensor, n_out: int):
            tensor_data = torch.cat((tensor_data, self.xavier_weight_initialization(1, n_out)), axis=0)
            momentum_tensor_data = torch.cat((momentum_tensor_data, torch.zeros(1, n_out, dtype=torch.float, device=MyDevice().get())), axis=0)
            return tensor_data, momentum_tensor_data

        if layer_number >= len(self.weight):
            [_, n_out] = self.output_weight.shape
            self.output_weight, self.output_momentum = add_element(self.output_weight, self.output_momentum, n_out)
        else:
            [_, n_out] = self.weight[layer_number].shape
            self.weight[layer_number], self.momentum[layer_number] = add_element(self.weight[layer_number], self.momentum[layer_number], n_out)

    def grow_weight_column(self, layer_number: int):
        def add_element(tensor_data: torch.tensor, momentum_tensor_data: torch.tensor, n_out: int):
            tensor_data = torch.cat((tensor_data, self.xavier_weight_initialization(n_out, 1)), axis=1)
            momentum_tensor_data = torch.cat((momentum_tensor_data, torch.zeros(n_out, 1, dtype=torch.float, device=MyDevice().get())), axis=1)
            return tensor_data, momentum_tensor_data

        if layer_number >= len(self.weight):
            [n_out, _] = self.output_weight.shape
            self.output_weight, self.output_momentum = add_element(self.output_weight, self.output_momentum, n_out)
        else:
            [n_out, _] = self.weight[layer_number].shape
            self.weight[layer_number], self.momentum[layer_number] = add_element(self.weight[layer_number], self.momentum[layer_number], n_out)

    def grow_bias(self, layer_number):
        def add_element(tensor_data: torch.tensor, momentum_tensor_data: torch.tensor, n_out: int):
            tensor_data = torch.cat((tensor_data, self.xavier_weight_initialization(1, n_out)), axis=1)
            momentum_tensor_data = torch.cat((momentum_tensor_data, torch.zeros(1, n_out, dtype=torch.float, device=MyDevice().get())), axis=1)
            return tensor_data, momentum_tensor_data

        if layer_number >= len(self.bias):
            [n_out, _] = self.output_bias.shape
            self.output_bias, self.output_bias_momentum = add_element(self.output_bias, self.output_bias_momentum, n_out)
        else:
            [n_out, _] = self.bias[layer_number].shape
            self.bias[layer_number], self.bias_momentum[layer_number] = add_element(self.bias[layer_number], self.bias_momentum[layer_number], n_out)
        pass

    def prune_node(self, layer_number: int, node_number: int):
        self.layers[layer_number] -= 1
        if layer_number >= 0:
            self.prune_weight_row(layer_number - 1, node_number)
            self.prune_bias(layer_number - 1, node_number)
        if layer_number <= self.number_hidden_layers:
            self.prune_weight_column(layer_number, node_number)

    def prune_weight_row(self, layer_number: int, node_number: int):
        def remove_nth_row(tensor_data: torch.tensor, n: int):
            return torch.cat([tensor_data[:n], tensor_data[n+1:]])

        if layer_number >= len(self.weight):
            self.output_weight = remove_nth_row(self.output_weight, node_number)
            self.output_momentum = remove_nth_row(self.output_momentum, node_number)
        else:
            self.weight[layer_number] = remove_nth_row(self.weight[layer_number], node_number)
            self.momentum[layer_number] = remove_nth_row(self.momentum[layer_number], node_number)

    def prune_weight_column(self, layer_number: int, node_number: int):
        def remove_nth_column(weight_tensor: torch.tensor, n: int):
            return torch.cat([weight_tensor.T[:n], weight_tensor.T[n+1:]]).T

        if layer_number >= len(self.weight):
            self.output_weight = remove_nth_column(self.output_weight, node_number)
            self.output_momentum = remove_nth_column(self.output_momentum, node_number)
        else:
            self.weight[layer_number] = remove_nth_column(self.weight[layer_number], node_number)
            self.momentum[layer_number] = remove_nth_column(self.momentum[layer_number], node_number)

    def prune_bias(self, layer_number: int, node_number: int):
        def remove_nth_element(bias_tensor: torch.tensor, n: int):
            bias_tensor = torch.cat([bias_tensor[0][:n], bias_tensor[0][n+1:]])
            return bias_tensor.view(1, bias_tensor.shape[0])

        if layer_number >= len(self.bias):
            self.output_bias = remove_nth_element(self.output_bias, node_number)
            self.output_bias_momentum = remove_nth_element(self.output_bias_momentum, node_number)
        else:
            self.bias[layer_number] = remove_nth_element(self.bias[layer_number], node_number)
            self.bias_momentum[layer_number] = remove_nth_element(self.bias_momentum[layer_number], node_number)