import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os


def load_all_isruc_S3(path, shuffle):
    read = pickle.load(open(path, 'rb'))
    psgs = read['psg']
    labels = read['labels']
    adjs = read['subject_adjs']

    psgs_concat = None
    adjs_concat = None
    labels_concat = None

    for i in range(len(psgs)):
        psgs_context, labels_context, adjs_context = add_context_single_sub(psgs[i], labels[i], adjs[i], context=1)

        if psgs_concat is None:
            psgs_concat, adjs_concat, labels_concat = psgs_context, adjs_context, labels_context
        else:
            psgs_concat = np.concatenate((psgs_concat, psgs_context))
            adjs_concat = torch.cat([adjs_concat, adjs_context])
            labels_concat = np.concatenate((labels_concat, labels_context))

    if shuffle:
        indexs = np.arange(len(psgs_concat))
        np.random.shuffle(indexs)
        psgs_concat = psgs_concat[indexs]
        adjs_concat = adjs_concat[indexs]
        labels_concat = labels_concat[indexs]

    return psgs_concat, adjs_concat, labels_concat

def add_context_single_sub(x, y, adj, context):
    """
    input:
        x       : eeg [batch,  channel, eeg]
        y       : [batch,  label]

        context : int

    return:
        x with contexts. [batch, context, channel, eeg]
        y with contexts. [batch, context, label]
    """
    # print(f'x shape {x.shape}')
    # print(f'y shape {y.shape}')
    adj = torch.from_numpy(adj)
    if context != 1:
        cut = context // 2
    else:
        return np.expand_dims(x, 1),y ,adj.unsqueeze(1).cpu()

    context_x = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
    context_adj = torch.zeros(x.shape[0] - 2 * cut, context, adj.shape[1], adj.shape[2])
    for i in range(cut, x.shape[0] - cut):
        context_x[i - cut] = x[i - cut:i + cut + 1]
        context_adj[i - cut] = adj[i - cut:i + cut + 1]

    context_y = y[cut:-cut]

    # print(f'x_c shape {context_x.shape}')
    # print(f'y_c shape {context_y.shape}')
    # print(f'context_adj shape {context_adj.shape}')

    return context_x, context_y, context_adj


def load_data_isruc_k_fold(psgs_concat, adjs_concat, labels_concat, train, K, i):
    """
    :param psgs_concat:
    :param adjs_concat:
    :param labels_concat:
    :param train:
    :param K:
    :param i:
    :return:
    """
    assert K > 1
    fold_size = psgs_concat.shape[0] // K  # 每份的个数:数据总条数/折数（组数）

    num_examples = len(psgs_concat)
    # print(f'examples number: {num_examples}')
    # print(f'adjs_concat: {adjs_concat.shape}')
    # print(f'features_concat: {psgs_concat.shape}')
    # print(f'label: {labels_concat.shape}')

    psgs_train, adjs_train, labels_train = None, None, None
    psgs_test, adjs_test, labels_test = None, None, None

    for j in range(K):
        idx = slice(j * fold_size, (j + 1) * fold_size)  

        psgs_part, adjs_part, labels_part = psgs_concat[idx], adjs_concat[idx], labels_concat[idx]
        if j == i:  
            # print(idx)
            psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
        elif psgs_train is None:
            psgs_train, adjs_train, labels_train = psgs_part, adjs_part, labels_part
        else:
            psgs_train = np.concatenate((psgs_train, psgs_part))  
            adjs_train = np.concatenate((adjs_train, adjs_part))
            labels_train = np.concatenate((labels_train, labels_part))

    if train:
        return psgs_train, adjs_train, labels_train
    else:
        return psgs_test, adjs_test, labels_test


class ISRUC_S3__subject_k_fold(Dataset):
    def __init__(self, data_path, num_context, K, i, train=True):
        super(ISRUC_S3__subject_k_fold, self).__init__()
        if train:
            self.trains, self.adjs, self.labels = get_subject_k_fold_isruc_S3(data_path, False, num_context, train=train, K=K, i=i)
        else:
            self.trains, self.adjs, self.labels = get_subject_k_fold_isruc_S3(data_path, False, num_context, train=train, K=K, i=i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.trains[index], self.adjs[index], self.labels[index]


def get_subject_k_fold_isruc_S3(path, shuffle, num_context, train, K, i):
    read = np.load(path, allow_pickle=True)
    psgs = read['psg']
    labels = read['labels']
    adjs = read['subject_adjs']

    psgs_train, adjs_train, labels_train = None, None, None
    psgs_test, adjs_test, labels_test = None, None, None

    for sub in range(K):
        psgs_context, labels_context, adjs_context = add_context_single_sub(psgs[sub], labels[sub], adjs[sub],
                                                                            context=num_context)


        psgs_part,  labels_part, adjs_part, = psgs_context, labels_context, adjs_context



        # mean = np.mean(psgs_part, 3)
        # psgs_part = psgs_part - mean.reshape((mean.shape[0],mean.shape[1],mean.shape[2], -1))

        if sub == i:  
            # print(idx)

            psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
            # psgs_test, adjs_test, labels_test = psgs_part, adjs_part, labels_part
        elif psgs_train is None:
            psgs_train, adjs_train, labels_train = psgs_part, adjs_part, labels_part
        else:
            psgs_train = np.concatenate((psgs_train, psgs_part)) 
            adjs_train = torch.cat([adjs_train, adjs_part])
            labels_train = np.concatenate((labels_train, labels_part))


    # adjs_concat = F.normalize(adjs_concat, p=1, dim=1)
    adjs_test = adjs_test != 0.  
    # adjs_concat = adjs_concat > 0  
    adjs_test = adjs_test.float()
    adjs_test += torch.eye(10)

    adjs_train = adjs_train != 0.  
    # adjs_concat = adjs_concat > 0  
    adjs_train = adjs_train.float()
    adjs_train += torch.eye(10)


    num_examples = len(psgs_train)
    if shuffle:
        indexs = np.arange(num_examples)

        np.random.shuffle(indexs)
        psgs_train = psgs_train[indexs]
        adjs_train = adjs_train[indexs]
        labels_train = labels_train[indexs]

    num_examples = len(psgs_test)
    if shuffle:
        indexs = np.arange(num_examples)

        np.random.shuffle(indexs)
        psgs_test = psgs_test[indexs]
        adjs_test = adjs_test[indexs]
        labels_test = labels_test[indexs]

    if train:
        return psgs_train, adjs_train, labels_train, 
    else:
        return psgs_test, adjs_test, labels_test



def get_subject_k_fold_data(k, i, batch_size, data_path, num_context):
    isruc_train = ISRUC_S3__subject_k_fold(data_path, num_context, k, i, train=True)
    isruc_test = ISRUC_S3__subject_k_fold(data_path, num_context, k, i, train=False)

    return (
        DataLoader(isruc_train, batch_size, shuffle=False, num_workers=0),
        DataLoader(isruc_test, batch_size, shuffle=False, num_workers=0)
    )

