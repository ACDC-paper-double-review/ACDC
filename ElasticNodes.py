import numpy as np


class ElasticNodes:
    growable = None
    prunable = None

    data_mean = 0
    data_standard_deviation = 0
    data_variance = 0

    number_samples_feed = 0
    number_samples_layer = None

    bias_mean = None
    bias_variance = None
    bias_standard_deviation = None
    minimum_bias_mean = None
    minimum_bias_standard_deviation = None
    bias = None

    var_mean = None
    var_variance = None
    var_standard_deviation = None
    minimum_var_mean = None
    minimum_var_standard_deviation = None
    var = None

    node_evolution = None

    bias_gradient = None
    bias_mean_net = None
    var_mean_net = None

    def __init__(self, number_hidden_layers=1):
        nhl = number_hidden_layers  # readability

        self.number_samples_layer = np.zeros(nhl)
        self.bias_mean = np.zeros(nhl)
        self.bias_variance = np.zeros(nhl)
        self.bias_standard_deviation = np.zeros(nhl)
        self.minimum_bias_mean = np.ones(nhl) * np.inf
        self.minimum_bias_standard_deviation = np.ones(nhl) * np.inf
        self.BIAS = []

        self.var_mean = np.zeros(nhl)
        self.var_variance = np.zeros(nhl)
        self.var_standard_deviation = np.zeros(nhl)
        self.minimum_var_mean = np.ones(nhl) * np.inf
        self.minimum_var_standard_deviation = np.ones(nhl) * np.inf
        self.VAR = []

        self.growable = np.ones(nhl) * False
        self.prunable = []

        for i in range(nhl):
            self.prunable.append([-1])



