import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from layers import BatchNorm
# from load_data import load_isruc_S3

from models import GCN, GraphConvStatic, GraphConvSTDP, GraphConvStdpWithAdj, StdpGCN, StdpGCN_test, StdpGCN_context

from utils import train, k_fold_train

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='validate during training pass')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=500, help="number of epochs to train")
parser.add_argument('--batch_size', type=int, default=256, help="number of batch")
parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
parser.add_argument('--weight_decay', type=float, default=5e-4, help="weight decay l2 loss on parameters")
parser.add_argument('--shuffle', type=bool, default=True, help='dropout rate 1-keep probability')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate 1-keep probability')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.stdpgcn = StdpGCN_context(nfeat=9, nid=256, nclass=9, dropout=0.5)
        self.flattern = nn.Flatten()
        self.fc1 = nn.Linear(450, 1024)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(216, 216)
        self.fc3 = nn.Linear(1024 + 216, 5)
    
    def forward(self, x, gril):
        x = self.stdpgcn(x)
        x = self.flattern(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        gril = gril.to("cuda:0").float()
        # gril = gril[:,:108]
        gril = self.fc4(gril)
        x = torch.cat((x, gril), dim=1)
        x = self.fc3(x)
        return x

# net = nn.Sequential(
#     # StdpGCN_test(nfeat=9, nid=256, nclass=9, dropout=0.5),
#     StdpGCN_context(nfeat=9, nid=256, nclass=9, dropout=0.5),
#     # StdpGCN(nfeat=9, nhid=256, nclass=9, dropout=0.5, eeg_size=3000),
#     # GraphConvStdpWithAdj(nfeat=9, nhid=256, nclass=32, dropout=0.5, eeg_size=3000),
#     # GraphConvSTDP(nfeat=18, nhid=256, nclass=32, dropout=0.5, eeg_size=1000),
#     # GraphConvStatic(nfeat=384, nhid=256, nclass=128, dropout=0.5, adj=adj),
#     # GraphConvStatic(nfeat=128, nhid=64, nclass=32, dropout=0.5, adj=adj),

#     nn.Flatten(),
#     nn.Linear(450, 1024), nn.ReLU(), nn.Dropout(p=0.5),
#     # # nn.Linear(512, 512), nn.ReLU(),nn.Dropout(p=0.5),
#     nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.5),
#     nn.Linear(1024, 5), nn.Softmax(dim=1)
#     # nn.Linear(90, 5), nn.Softmax()
# )

net = Model()

X = torch.randn([1, 5, 10, 9])
adjs = torch.randn([1, 5, 10, 10])

# print(net((X, adjs)).shape)
# for layer in net:
#     if isinstance(layer, StdpGCN_context):
#         X = layer((X, adjs))
#     else:
#         X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)
# wd = net[2].weight.norm().item()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)
net = net.to("cuda:0")
# input = torch.randn([50, 10, 18])
# print(net(input).shape)

# 不k折交叉验证训练程序
# train_iter, test_iter = load_isruc_S3(batch_size=args.batch_size,
#                                       path="D:\\data\\ISRUC_S3\\ISRUC_S3_features_adjs_re.npz", shuffle=True)
# train(net, train_iter, test_iter, args.epochs, args.lr, device="cuda:0")

k_fold_train(net, 10, num_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device="cuda:0",
             shuffle=args.shuffle)
