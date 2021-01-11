import math
import torch


class MyUtil:
    def __init__(self):
        pass

    @staticmethod
    def recursive_mean_standard_deviation(x, old_mean, old_variance, number_samples):
        mean = old_mean + (x - old_mean) / number_samples
        var = old_variance + (x - old_mean) * (x - mean)
        return mean, var, torch.sqrt(var/number_samples)

    @staticmethod
    def probit(mean, standard_deviation):
        p = (1 + math.pi * (standard_deviation ** 2) / 8)
        return mean / torch.sqrt(p)

    @staticmethod
    def norm_1(x):
        return torch.norm(x, 1)

    @staticmethod
    def norm_2(x):
        return torch.norm(x, 2)

    @staticmethod
    def frobenius_norm(x):
        return torch.norm(x, 'fro')