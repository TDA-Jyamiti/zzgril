import torch
import torch.utils.data
from torch.utils.data import TensorDataset
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_default_train_val_test_loader(args):

    # get dataset-id
    dsid = args.dataset

    gril_graphs = args.gril_graphs
    num_center_pts = args.num_center_pts
    # get dataset from .pt
    data_train = torch.load(f'../data/UCR/{dsid}/X_train.pt')
    data_val = torch.load(f'../data/UCR/{dsid}/X_val.pt')
    label_train = torch.load(f'../data/UCR/{dsid}/y_train.pt')
    label_val = torch.load(f'../data/UCR/{dsid}/y_val.pt')
    
    # get gril 
    
    if gril_graphs:
        gril_train_l0 = np.load(f'../saved_zz_gril/zz_graphs/UCR_time_series/{dsid}/{num_center_pts}_l0_train.npy')
        gril_train_l1 = np.load(f'../saved_zz_gril/zz_graphs/UCR_time_series/{dsid}/{num_center_pts}_l1_train.npy')
        gril_val_l0 = np.load(f'../saved_zz_gril/zz_graphs/UCR_time_series/{dsid}/{num_center_pts}_l0_test.npy')
        gril_val_l1 = np.load(f'../saved_zz_gril/zz_graphs/UCR_time_series/{dsid}/{num_center_pts}_l1_test.npy')
        
    else:
        gril_train_l0 = np.load(f'../saved_zz_gril/zz_point_clouds/UCR_time_series/{dsid}_{num_center_pts}_center_pts/processed_train_l0.npy')
        gril_train_l1 = np.load(f'../saved_zz_gril/zz_point_clouds/UCR_time_series/{dsid}_{num_center_pts}_center_pts/processed_train_l1.npy')
        gril_val_l0 = np.load(f'../saved_zz_gril/zz_point_clouds/UCR_time_series/{dsid}_{num_center_pts}_center_pts/processed_test_l0.npy')
        gril_val_l1 = np.load(f'../saved_zz_gril/zz_point_clouds/UCR_time_series/{dsid}_{num_center_pts}_center_pts/processed_test_l1.npy')
    
    num_nodes = data_val.size(-2)
    seq_length = data_val.size(-1)
    
    num_classes = len(torch.bincount(label_val.type(torch.int)))
    
    gril_train_l0 = gril_train_l0.reshape(-1, 108)
    gril_train_l1 = gril_train_l1.reshape(-1, 108)
    gril_val_l0 = gril_val_l0.reshape(-1, 108)
    gril_val_l1 = gril_val_l1.reshape(-1, 108)
    
    if args.use_only_lambda_0:
        gril_train = gril_train_l0
        gril_val = gril_val_l0
    else:
        gril_train = np.hstack([gril_train_l0, gril_train_l1])
        gril_val = np.hstack([gril_val_l0, gril_val_l1])
    
    # convert data & labels to TensorDataset
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_val, label_val)

    # data_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=args.workers, 
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=args.val_batch_size, 
                                             shuffle=False, 
                                             num_workers=args.workers, 
                                             pin_memory=True)
    
    gril_train_dataset = TensorDataset(torch.tensor(gril_train), label_train)
    gril_val_dataset = TensorDataset(torch.tensor(gril_val), label_val)
    
    gril_train_loader = torch.utils.data.DataLoader(gril_train_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=False, 
                                                   num_workers=args.workers, 
                                                   pin_memory=True)
    
    gril_val_loader = torch.utils.data.DataLoader(gril_val_dataset,
                                                    batch_size=args.val_batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    


    return train_loader, val_loader, num_nodes, seq_length, num_classes, gril_train_loader, gril_val_loader
