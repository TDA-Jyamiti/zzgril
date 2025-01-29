# def stdp(pre_t, previous_post_t, next_post_t, delta_ext):
#     """
#     :param pre_t: [batch, ]
#     :param previous_post_t:
#     :param next_post_t:
#     :param delta_ext:
#     :return:
#     """
#     # if next_post_spike_t - pre_spike_t < pre_spike_t - previous_post_spike_t:
#     #     post_spike_t = next_post_spike_t
#     # else:
#     #     post_spike_t = previous_post_spike_t
#     # delta_t = post_spike_t - pre_spike_t
#     # if delta_t != 0 and np.abs(delta_t) != np.Inf:
#     #     return np.abs(delta_ext) / delta_t
#     # return 0
import math
import multiprocessing
from time import time

import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from STDP.encoder import load_psg, norm_eeg, BSA
# from load_data import load_isruc_S3
from STDP.stdp_weight import STDP

from sklearn.preprocessing import normalize
import numpy as np
from numba import cuda
import numba


class GraphConstructorMulti:
    def __init__(self, adj_size):
        self.stdp = STDP()
        self.adj = np.zeros((adj_size))

    def get_graph(self, input, i):
        """
        单个图的生成
        并发
        :param input:[channel_num, eeg]
        :return: no return
        """

        channel_num, eeg_length = input.shape
        # adj = torch.eye(channel_num).float()

        pre = input
        post = input

        # print(pre.device)
        # print(f'pre shape : {pre.t().shape} post shape: {post.t().shape}')

        w_adj = self.stdp.get_stdp_weight_multi(pre.t(), post.t(), channel_num=channel_num)
        w_adj = normalize(w_adj.cpu(), axis=0, norm='max')
        w_adj += np.identity(w_adj.shape[0])

        self.adj[i] = w_adj


class GraphConstructSTDP:
    """
    这里的设计需要转变
    设计是
        输入 -> GraphConvSTDP（图卷积类 在图卷积类中生成邻接矩阵再进行图卷积计算） -> 输出图卷积后的图
    本类是在 GraphConvSTDP 中实现构建邻接矩阵的方法
    可以改成函数
    """

    def __init__(self, device):
        super(GraphConstructSTDP, self).__init__()
        self.adj = torch.eye(0).float()
        self.device = device

    def get_batch_graph_test(self, input):
        """
               :param eeg: [batch, channel, encoded eeg]
               :return: [batch, adj]
               """
        # TODO    最简单的写法，效率最低的写法 应该有并行实现的方法

        # adjs = []
        adjs = torch.zeros(input.shape[0], input.shape[1], input.shape[1]).to(self.device)
        eye = torch.eye(input.shape[1]).to(self.device)

        # print(self.device)
        pbar = tqdm(input)
        for i, epoch in enumerate(pbar):
            pbar.set_description(f'STDP adj getting epoch:')
            # for epoch in input:
            # temp_adj = self.one_graph_construct(epoch)
            # start = time()

            temp_adj = self.one_graph_construct_multi(epoch)
            # print("one_graph_construct_multi " + str(time() - start))
            # TODO 这里对矩阵正则化 已经转到读取数据的时候
            # TODO 这里全部做成gpu代码，不在cpu这里切换 已经完成
            # TODO 这里已经成为预处理的一部分，正则化和预处理需要在训练过程中实现
            # temp_adj = normalize(temp_adj.cpu(), axis=0, norm='max')
            # start = time()
            # temp_adj = F.normalize(temp_adj, p=1, dim=1)
            # print("F.normalize " + str(time() - start))

            # temp_adj += np.identity(temp_adj.shape[0])
            # start = time()
            # temp_adj += eye
            # print("eye " + str(time() - start))

            adjs[i] = temp_adj

        self.adj = adjs
        # return torch.tensor(adjs).double().to(self.device)
        return adjs

    def one_graph_construct_multi(self, input):
        """
        TODO 这里生成的图是上面那个one方法生成的图的转置，至于什么影响需要实验先看看情况
        :param input:
        :return:
        """
        channel_num, eeg_length = input.shape
        # adj = torch.eye(channel_num).float()

        pre = input
        post = input

        stdp = STDP()
        # print(pre.device)
        # print(f'pre shape : {pre.t().shape} post shape: {post.t().shape}')

        w_adj = stdp.get_stdp_weight_multi(pre.t(), post.t(), channel_num=channel_num)
        # print(w_adj)

        return w_adj


if __name__ == '__main__':
    # train_iter, test_iter = load_isruc_S3(batch_size=20, path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_origin.npz")
    adjs_constructor = GraphConstructSTDP(device="cuda:0")
    # for x, y in train_iter:
    #     print(x.shape)
    #     print(y.shape)
    #     adjs = adjs_constructor.one_graph_construct_cuda(input=x[0, :, :3000])
    #     print(adjs.shape)
    #     print(adjs[0])
    #     break

    x = torch.randn(128, 10, 500)
    print(x.shape)
    x = x.to("cuda:0")
    start = time()
    # adjs = adjs_constructor.get_batch_graph_multiprocessing(input=x)
    adjs = adjs_constructor.get_batch_graph(input=x)
    print("multiing adj time " + str(time() - start))
    start = time()
    # # adjs_2 = adjs_constructor.one_graph_construct(input=x)
    adjs_2 = adjs_constructor.get_batch_graph_test(input=x)
    print("no multing adj time " + str(time() - start))

    # print(adjs_2 == adjs)
    # print(adjs)
