import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.nn.modules.module import Module


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta

    return Y, moving_mean.data, moving_var.data


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta,
            self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9
        )

        return Y


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() 
    def reset_parameters(self):
        # nn.init.xavier_normal_(self.weight)
        # if self.bias is not None:
        #     nn.init.xavier_normal_(self.bias)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(f'1 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        # self.weight.data = self.weight.data.unsqueeze(0)
        # self.weight.data = self.weight.data.repeat(input.shape[0], 1, 1)
        # print(f'1 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        # support = torch.bmm(input, self.weight)  
        # output = torch.bmm(adj, support)  
        support = input @ self.weight 
        # print(f'2 support shape {support}, adj shape{adj.shape} self weight {self.weight.shape}')
        # support = support.to(torch.float32)
        # adj = adj.to(torch.float32)
        output = adj @ support  
        # print(f'2 support shape {support.shape}, adj shape{adj.shape} output {output.shape}')
        # self.weight.data = self.weight.data[0]
        # print(f'3 input shape {input.shape}, adj shape{adj.shape} self weight {self.weight.shape}')

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


if __name__ == '__main__':
    input = torch.arange(0, 4).resize(2, 2).float()
    input = input.repeat(2, 1, 1)
    adj = torch.ones(2, 2)
    gcn = GraphConvolution(in_features=2, out_features=2)
    print(gcn(input, adj))
