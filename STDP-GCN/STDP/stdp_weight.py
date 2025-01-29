import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer, neuron, functional
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

# from load_data import load_isruc_S3


class STDP:

    def __init__(self, pre_tau=100., post_tau=100.):
        self.pre_spikes = []
        self.post_spikes = []
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.trace_pre = []
        self.trace_post = []
        self.w = []

    def get_trace_stdp_weight(self, pre_spikes, post_spikes, lr=1e-2):
        self.pre_spikes = pre_spikes
        self.post_spikes = post_spikes

        fc = nn.Linear(1, 1, bias=False).to(pre_spikes.device)
        fc.weight.data = torch.tensor([[0.0]])
        # print(f'weight init {fc.weight.item()}')

        stdp_learner = layer.STDPLearner(self.pre_tau, self.post_tau, self.f_pre, self.f_post)
        if len(self.pre_spikes) == len(self.post_spikes):
            T = len(self.pre_spikes)
            # for t in tqdm(range(T)):
            # print(self.pre_spikes.device)
            for t in range(T):
                stdp_learner.stdp(self.pre_spikes[t], self.post_spikes[t], fc, lr)
                # self.trace_pre.append(stdp_learner.trace_pre.item())
                # self.trace_post.append(stdp_learner.trace_post.item())
                self.w.append(fc.weight.item())
        else:
            raise NotImplementedError

        return fc.weight.item()

    def get_stdp_weight_multi(self, pre_spikes, post_spikes, channel_num=2, lr=1e-2, ):
        self.pre_spikes = pre_spikes

        self.post_spikes = post_spikes

        # print(f'self.pre_spikes shape {self.pre_spikes.shape}')
        fc = nn.Linear(channel_num, channel_num, bias=False).to(pre_spikes.device)

        # print(f'weight init {fc.weight.data.shape}')
        fc.weight.data = torch.zeros_like(fc.weight.data)
        # print(f'weight init {fc.weight.item()}')

        stdp_learner = layer.STDPLearner(self.pre_tau, self.post_tau, self.f_pre, self.f_post)
        if len(self.pre_spikes) == len(self.post_spikes):
            T = len(self.pre_spikes)

            for t in range(T):
                stdp_learner.stdp(self.pre_spikes[t], self.post_spikes[t], fc, lr)

        else:
            raise NotImplementedError

        return fc.weight.data

    # F+(wij),F−(wij)
    def f_pre(self, x):
        return x.abs() + 0.1

    def f_post(self, x):
        return - self.f_pre(x)


