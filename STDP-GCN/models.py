import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from STDP.STDP_graph import GraphConstructSTDP
from layers import GraphConvolution


class StdpGCN_context(nn.Module):

    def __init__(self, nfeat, nid, nclass, dropout):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass: 
        :param eeg_size:
        :param dropout:
        """
        super(StdpGCN_context, self).__init__()
        self.dropout = dropout
        self.gcn = GCN(nfeat, nid, nclass, dropout)
        self.bn = nn.BatchNorm2d(9)

        self.time_conv = nn.Conv2d(in_channels=nfeat, out_channels=nclass, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(0, 1))

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features],
                      adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input  

        # adj = torch.randn(adjs.shape) > 0
        ones_adj = torch.ones(adjs.shape).float().to(features.device)
        # random_adj = adj.float().to(features.device)
        # print(adj[0])

        timesteps = features.shape[1]
        x = torch.zeros_like(features)
        for t in range(1, timesteps):
            # x[:, t, :, :] = self.gcn(features[:, t, :, :], adjs[:, t, :, :])
            # x[:, t, :, :] = self.gcn(features[:, t, :, :], random_adj[:, t, :, :])  
            x[:, t, :, :] = self.gcn(features[:, t, :, :], ones_adj[:, t, :, :]) 

        # print(f'context GCN shape{x.shape}')
        # output shape [batch, 1, channel, features']
        x_time = rearrange(x, 'b c h w -> b w h c')
        # print(f'rearrange shape{x_time.shape}')

        x_time_conv = self.time_conv(x_time)
        x_time_conv = self.bn(x_time_conv)
        # print(f'x_time_conv shape{x_time_conv.shape}')
        x_time_conv = rearrange(x_time_conv, 'b w h c -> b c h w')
        # print(f'x_time_conv rearrange shape{x_time_conv.shape}')

        # x_residual = self.short_cut(x)
        # print(f'x_residual shape{x_residual.shape}')
        output = features + x_time_conv

        # return F.log_softmax(x_time_conv, dim=1)
        return output


class StdpGCN_test(nn.Module):

    def __init__(self, nfeat, nid, nclass, dropout):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass: 
        :param eeg_size:
        :param dropout:
        """
        super(StdpGCN_test, self).__init__()
        self.dropout = dropout
        self.gcn = GCN(nfeat, nid, nclass, dropout)

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features], adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input

        # output shape [batch,  channel, features]
        x = self.gcn(features, adjs)

        return x


class StdpGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
                :param nfeat: 
                :param nhid: 
                :param nclass: 
                :param eeg_size: 
                :param dropout:
                """
        super(StdpGCN, self).__init__()
        self.eeg_size = eeg_size
        self.gcn = GCN(nfeat, nhid, nclass, dropout)
        self.time_conv = nn.Conv2d(in_channels=nfeat, out_channels=nclass, kernel_size=(1, 3), stride=(1, 1),
                                   padding=(1, 1))
        self.short_cut = nn.Conv2d(in_channels=nfeat, out_channels=5, kernel_size=1, stride=1, padding=(0, 1))

    def forward(self, input):
        """

        :param input: x=[batch,context, channel, encoded eeg + features], adjs=[batch, channel, channel]
        :return:
        """
        features, adjs = input  
        # features = x[:, 2, :, :]
        # adjs = adjs[:, 2, :, :]

        # output shape [batch, context, channel, features]
        x = self.gcn(features, adjs)

        # print(f'context GCN shape{x.shape}')
        # output shape [batch, 1, channel, features']
        # x_time = rearrange(x, 'b c h w -> b w h c')
        # print(f'rearrange shape{x_time.shape}')

        # x_time_conv = self.time_conv(x_time)
        # x_time_conv = rearrange(x_time_conv, 'b w h c -> b c h w')
        # print(f'x_time_conv shape{x_time_conv.shape}')

        # x_residual = self.short_cut(x)
        # print(f'x_residual shape{x_residual.shape}')

        # output = x_residual + x_time_conv

        # return F.log_softmax(x_time_conv, dim=1)
        return x


class GraphConvStdpWithAdj(nn.Module):


    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass: 
        :param eeg_size: 
        :param dropout:
        """
        super(GraphConvStdpWithAdj, self).__init__()
        self.eeg_size = eeg_size

        # self.adj_by_STDP = []
        #
        # self.adjs_constructor = GraphConstructSTDP(device="cuda:0")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, input):
        """
        :param input: x=[batch, epoch, channel, encoded eeg +features], adjs=[batch, channel, channel]
        :return:
        """
        x, adjs = input
        # print(adjs)
        with torch.no_grad():
            batch_size = x.shape[0]
            eeg = x[:, :, :self.eeg_size]
            features = x[:, :, :, self.eeg_size + 9:]
            # print(f'eeg shape : {features.shape}')
            # print(f'x device : {x.device}')

            self.adj_by_STDP = adjs
            # self.adj_by_STDP = self.adjs_constructor.get_batch_graph(input=eeg).to(torch.float32)
            # self.adj_by_STDP = adjs_constructor.get_batch_graph_multiprocessing(input=eeg).to(torch.float32)

        x = F.relu(self.gc1(features, self.adj_by_STDP))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj_by_STDP)
        # print(x.shape)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GraphConvSTDP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, eeg_size):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass: 
        :param eeg_size: 
        :param dropout:
        """
        super(GraphConvSTDP, self).__init__()
        self.eeg_size = eeg_size
        self.adj_by_STDP = []

        self.adjs_constructor = GraphConstructSTDP(device="cuda:0")
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        """
        :param x: [batch, epoch, channel, encoded eeg +features]
        :return:
        """

        with torch.no_grad():
            batch_size = x.shape[0]
            eeg = x[:, :, :self.eeg_size]
            features = x[:, :, self.eeg_size:]
            # print(f'eeg shape : {features.shape}')
            # print(f'x device : {x.device}')

            self.adj_by_STDP = self.adjs_constructor.get_batch_graph(input=eeg).to(torch.float32)
            # self.adj_by_STDP = adjs_constructor.get_batch_graph_multiprocessing(input=eeg).to(torch.float32)

        x = F.relu(self.gc1(features, self.adj_by_STDP)) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj_by_STDP)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GraphConvStatic(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout, adj):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass: 
        :param dropout:
        """
        super(GraphConvStatic, self).__init__()
        self.adj = adj

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x):
        batch_size = x.shape[0]
        # self.adj = self.adj.repeat(batch_size, 1, 1)
        x = F.relu(self.gc1(x, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)
        # print(x.shape)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        :param nfeat: 
        :param nhid: 
        :param nclass:
        :param dropout:
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout

    def forward(self, x, adj):
        # print(f'x device{x.device}')
        # print(f'adj device{adj.device}')

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    adj = torch.ones((2, 10, 10))
    net = nn.Sequential(
        # GraphConvStatic(nfeat=18, nhid=20, nclass=18, dropout=0.5, adj=adj), nn.ReLU(),
        # GraphConvSTDP(nfeat=18, nhid=20, nclass=18, dropout=0.5, eeg_size=1000), nn.ReLU(),
        # StdpGCN(nfeat=9, nhid=20, nclass=9, dropout=0.5, eeg_size=3000), nn.ReLU(),
        # StdpGCN_test(nfeat=9, nclass=9, dropout=0.5),
        StdpGCN_context(nfeat=9, nid=256, nclass=9, dropout=0.5),
        nn.Flatten(), nn.ReLU(),
        nn.Linear(90, 5)
    )

    # context
    input = torch.randn([2, 5, 10, 9])
    input_adj = torch.randn([2, 5, 10, 10])
    # no context
    # input = torch.randn([2, 10, 9])
    # input_adj = torch.randn([2, 10, 10])
    input = (input, input_adj)
    y = torch.randint(0, 4, [2])

    output = net(input)
    print(output.shape)
