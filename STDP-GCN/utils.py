import os
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader

from visdom import Visdom

import layers
from load_data_subject_K_fold import load_all_isruc_S3, get_subject_k_fold_data

import matplotlib.pyplot as plt

# viz = Visdom(env="graph stdp k fold")


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_f1(net, data_iter, gril_test, device="cuda:0"):

    prob_all = []
    label_all = []
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    for (X, adj, y), gril, in zip(data_iter, gril_test):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        X = X.to(torch.float32)
        adj = adj.to(device)
        y = y.to(device)

        prob = net((X, adj), gril)
        prob = prob.cpu().detach().numpy()
        prob_all.extend(np.argmax(prob, axis=1))
        label_all.extend(y.cpu().numpy())

    return f1_score(label_all, prob_all,average='macro'), f1_score(label_all, prob_all,average=None)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy_gpu(net, data_iter, gril_test, device="cuda:0"):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = Accumulator(2)
    for (X, adj, y), gril, in zip(data_iter, gril_test):
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        X = X.to(torch.float32)
        adj = adj.to(device)
        y = y.to(device)
        metric.add(accuracy(net((X, adj), gril), y), y.numel())

    return metric[0] / metric[1]


def k_fold_train(net, k, num_epochs, lr, batch_size, device, shuffle, num_center_pts):
    train_acc_sum, test_acc_sum, test_f1_sum = 0, 0, 0

    for i in range(k):
        # data = get_k_fold_data(k, i, X_train, y_train)
        psgs_concat, adjs_concat, labels_concat = load_all_isruc_S3(
            path="./data/ISRUC_S3/ISRUC_S3_features_adjs_re.pkl", shuffle=shuffle)
        train_iter, test_iter = get_subject_k_fold_data(k, i, batch_size, "./data/ISRUC_S3/ISRUC_S3_features_adjs_re.pkl", 5)
        gril_path = '../saved_zz_gril/zz_graphs/ISRUC_S3/{num_center_pts}_center_pts'
        gril_l0_train = np.zeros((1,108))
        gril_l1_train = np.zeros((1,108))
        gril_l0_test = np.zeros((1,108))
        gril_l1_test = np.zeros((1,108))
        
        for file in os.listdir(gril_path):
            if int(file.split('_')[3].split('.')[0]) == i:
                if 'l0' in file:
                    l0 = np.load(os.path.join(gril_path, file)).reshape(-1,108)
                    gril_l0_test = np.concatenate((gril_l0_test, l0[:-4,:]), axis=0)
                elif 'l1' in file:
                    l1 = np.load(os.path.join(gril_path, file)).reshape(-1,108)
                    gril_l1_test = np.concatenate((gril_l1_test, l1[:-4,:]), axis=0)
            else:
                if 'l0' in file:
                    l0 = np.load(os.path.join(gril_path, file)).reshape(-1,108)
                    gril_l0_train = np.concatenate((gril_l0_train, l0[:-4,:]), axis=0)
                elif 'l1' in file:
                    l1 = np.load(os.path.join(gril_path, file)).reshape(-1,108)
                    gril_l1_train = np.concatenate((gril_l1_train, l1[:-4,:]), axis=0)
        gril_l0_train = gril_l0_train[1:,:]
        gril_l1_train = gril_l1_train[1:,:]
        gril_l0_test = gril_l0_test[1:,:]
        gril_l1_test = gril_l1_test[1:,:]
        gril_train = np.concatenate((gril_l0_train, gril_l1_train), axis=1)
        gril_test = np.concatenate((gril_l0_test, gril_l1_test), axis=1)
        
        gril_train = DataLoader(gril_train, batch_size, shuffle=False, num_workers=0)
        gril_test = DataLoader(gril_test, batch_size, shuffle=False, num_workers=0)
    
        train_acc, test_acc, test_f1 = train(net, train_iter, test_iter, num_epochs, lr, device=device, 
                                    gril_train=gril_train, gril_test=gril_test, k=k, k_i=i)

        print('train_loss:%.6f' % train_acc, 'test_acc:%.4f\n' % test_acc)

        train_acc_sum += train_acc
        test_acc_sum += test_acc
        test_f1_sum += test_f1
        
    print('train_acc_avg:%.4f\n' % (train_acc_sum / k), 'test_acc_avg:%.4f' % (test_acc_sum / k), 
          'test_f1_avg:%.4f' % (test_f1_sum / k))


def train(net, train_iter, test_iter, num_epochs, lr, device, gril_train, gril_test, k=0, k_i=0):
    """
    :param net:
    :param train_iter: 
    :param test_iter: 
    :param num_epochs:
    :param lr: 
    :param device: gpu
    :return:
    """

    best_train_acc = 0
    best_test_acc = 0
    
    best_test_f1 = 0
    best_class_f1s = [0,0,0,0,0]

    def init_weights(m):
        # print(1, type(m))
        if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == layers.GraphConvolution:
            # print(2, type(m))
            nn.init.xavier_normal_(m.weight)

    net.apply(init_weights)

    # num_batches = len(train_iter)
    print('training on', device)


    net.to(device)

    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # loss = gl_loss()

    # train
    cnt = 0
    for epoch in range(num_epochs):
        net.train()

        metric = Accumulator(3)
        train_acc = []
        for i, (X, adj, y) in enumerate(train_iter):
            # X, y = X.to(device), y.to(device)
            # plt.imshow(X[0][0])
            # plt.show()
            X, adj, y = X.float().to(device), adj.to(device), y.to(device)
            optimizer.zero_grad()
            X = X.float().to(device)
            gril_iter = iter(gril_train)
            gril = next(x for j,x in enumerate(gril_iter) if j == i)
            y_hat = net((X, adj), gril)

            l = loss(y_hat, y.long())

            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y.long()), X.shape[0])

            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            cnt += 1
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            # viz.line([[train_acc, train_l]], [cnt], win='loss', update='append')

            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     print(f'epoch: {epoch} train loss: {train_l}, train_acc {train_acc}')
            # print(f'epoch: {epoch} train loss: {train_l}, train_acc {0}')

        test_acc = evaluate_accuracy_gpu(net, test_iter, gril_test)
        test_f1, class_f1s = evaluate_f1(net, test_iter, gril_test)
        
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_class_f1s = class_f1s
        # viz.line([[train_acc, test_acc]], [epoch], win=f'acc {k} fold {k_i}', update='append')
        print(f'epoch: {epoch} loss {train_l:.3f}, train acc {train_acc:.3f}',
              f'test acc {test_acc:.3f}',
              f'best f1 {best_test_f1:.3f}',
              f'best acc {best_test_acc:.3f}',
              f'best class f1s {best_class_f1s}')
            #   f'on {str(device)}')
        # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

    return best_train_acc, best_test_acc, best_test_f1
