from NeuralNetwork import NeuralNetwork
from MySingletons import MyDevice

import numpy as np
import torch


class AutoEncoder(NeuralNetwork):
    _greedy_layer_bias = None
    _greedy_layer_output_bias = None

    @property
    def latent_space(self):
        return self.layer_value[self.latent_space_position]

    @property
    def latent_space_size(self):
        return self.layers[self.latent_space_position]

    @property
    def latent_space_position(self):
        return int((len(self.layers) - 1) / 2)

    def __init__(self, layers=[]):
        NeuralNetwork.__init__(self, layers)
        for i in range(self.number_hidden_layers):
            self.activation_function[i] = self.ACTIVATION_FUNCTION_SIGMOID
        self.output_activation_function = self.ACTIVATION_FUNCTION_SIGMOID
        self.loss_function = self.LOSS_FUNCTION_MSE

    def train(self, x: torch.tensor, is_tied_weight: bool = False, noise_ratio: float = 0.0, weight_number: int = None, y: torch.tensor = None):
        if is_tied_weight:
            for i in range(int(self.number_hidden_layers/2)):
                if i == 0:
                    self.output_weight = self.weight[i].T
                else:
                    self.weight[-i] = self.weight[i].T

        if y is None:
            y = x
        NeuralNetwork.train(self, x=self.masking_noise(x=x, noise_ratio=noise_ratio), y=y, weight_no=weight_number)

    def test(self, x: torch.tensor, is_beta_updatable: bool = False, y: torch.tensor = None):
        if y is None:
            y = x
        return NeuralNetwork.test(self, x=x, y=y, is_beta_updatable=is_beta_updatable)

    def grow_node(self, layer_number):
        NeuralNetwork.grow_node(self, layer_number)
        self.grow_greedy_layer_bias(layer_number)

    def prune_node(self, layer_number, node_number):
        NeuralNetwork.prune_node(self, layer_number, node_number)
        self.prune_greedy_layer_bias(layer_number, node_number)

    def grow_greedy_layer_bias(self, layer_number):
        b = layer_number
        if b is self.number_hidden_layers:
            [n_out, n_in] = self._greedy_layer_output_bias.shape
            self._greedy_layer_output_bias = torch.cat((self._greedy_layer_output_bias, self.xavier_weight_initialization(1, 1)), axis=1)
        else:
            [n_out, n_in] = self._greedy_layer_bias[b].shape
            n_in = n_in + 1
            self._greedy_layer_bias[b] = np.append(self._greedy_layer_bias[b], self.xavier_weight_initialization(n_out, n_in, shape=(n_out, 1)))

    def grow_layer(self, option, number_of_nodes):
        raise TypeError('Not implemented')

    def prune_greedy_layer_bias(self, layer_number, node_number):
        def remove_nth_element(greedy_bias_tensor, n):
            bias_tensor = torch.cat([greedy_bias_tensor[0][:n], greedy_bias_tensor[0][n + 1:]])
            return bias_tensor.view(1, bias_tensor.shape[0])

        b = layer_number  # readability
        n = node_number  # readability

        if b is self.number_hidden_layers:
            self._greedy_layer_output_bias = remove_nth_element(self._greedy_layer_output_bias, n)
        else:
            self._greedy_layer_bias[b] = remove_nth_element(self._greedy_layer_bias[b], n)

    def greedy_layer_wise_pretrain(self, x: torch.tensor, number_epochs: int = 1, is_tied_weight: bool = False,
                                   noise_ratio: float = 0.0):
        for i in range(len(self.layers) - 1):
            if i > self.number_hidden_layers:
                nn = NeuralNetwork([self.layers[i], self.layers[-1], self.layers[i]], init_weights=False)
            else:
                nn = NeuralNetwork([self.layers[i], self.layers[i + 1], self.layers[i]], init_weights=False)

            nn.activation_function[0] = nn.ACTIVATION_FUNCTION_SIGMOID
            nn.output_activation_function = nn.ACTIVATION_FUNCTION_SIGMOID
            nn.loss_function = nn.LOSS_FUNCTION_MSE
            nn.momentum_rate = 0

            if i >= self.number_hidden_layers:
                nn.weight[0] = self.output_weight.clone()
                nn.bias[0] = self.output_bias.clone()
                nn.output_weight = self.output_weight.T.clone()
                if self._greedy_layer_output_bias is None:
                    nodes_after = nn.layers[-1]

                    self._greedy_layer_output_bias = self.xavier_weight_initialization(1, nodes_after)
                nn.output_bias = self._greedy_layer_output_bias.clone()
            else:
                nn.weight[0] = self.weight[i].clone()
                nn.bias[0] = self.bias[i].clone()
                nn.output_weight = self.weight[i].T.clone()
                try:
                    nn.output_bias = self._greedy_layer_bias[i].detach()
                except (TypeError, IndexError):
                    nodes_after = nn.layers[-1]

                    if self._greedy_layer_bias is None:
                        self._greedy_layer_bias = []

                    self._greedy_layer_bias.append(self.xavier_weight_initialization(1, nodes_after))
                    nn.output_bias = self._greedy_layer_bias[i].clone()

            for j in range(0, number_epochs):
                training_x = self.forward_pass(x=x).layer_value[i].detach()
                nn.train(x=self.masking_noise(x=training_x, noise_ratio=noise_ratio), y=training_x)

            if i >= self.number_hidden_layers:
                self.output_weight = nn.weight[0].clone()
                self.output_bias = nn.bias[0].clone()
            else:
                self.weight[i] = nn.weight[0].clone()
                self.bias[i] = nn.bias[0].clone()

    def update_weights_kullback_leibler(self, Xs, Xt, gamma=0.0001):
        loss = NeuralNetwork.update_weights_kullback_leibler(self, Xs, Xs, Xt, Xt, gamma)
        return loss

    def compute_evaluation_window(self, x):
        raise TypeError('Not implemented')

    def compute_bias(self, y):
        return torch.mean((self.Ey.T - y) ** 2)

    @property
    def network_variance(self):
        return torch.mean(self.Ey2 - self.Ey ** 2)


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, layers=[]):
        AutoEncoder.__init__(self, layers)
        # FIXME: The lines below are just to build the greedy_layer_bias. Find a more intuitive way to perform it
        random_x = np.random.rand(layers[0])
        random_x = torch.tensor(np.atleast_2d(random_x), dtype=torch.float, device=MyDevice().get())
        self.greedy_layer_wise_pretrain(x=random_x, number_epochs=0)

    def train(self, x: torch.tensor, noise_ratio: float = 0.0, is_tied_weight: bool = False, weight_number: int = None, y: torch.tensor = None):
        AutoEncoder.train(self, x=x, noise_ratio=noise_ratio, is_tied_weight=is_tied_weight, weight_number=weight_number, y=y)

    def greedy_layer_wise_pretrain(self, x: torch.tensor, number_epochs: int = 1, is_tied_weight: bool = False, noise_ratio: float = 0.0, y: torch.tensor = None):
        AutoEncoder.greedy_layer_wise_pretrain(self, x=x, number_epochs=number_epochs, is_tied_weight=is_tied_weight, noise_ratio=noise_ratio)