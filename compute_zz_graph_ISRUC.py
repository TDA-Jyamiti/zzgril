import numpy as np
from model import zzMultipersGraph
import os
from argparse import ArgumentParser
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import torch
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ISRUC_S3')
    parser.add_argument('--num_center_pts', type=int, default=36)
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    num_center_pts = args.num_center_pts
    fold = args.fold
    dataset = args.dataset
    
    edge_list_data = pickle.load(open(f'data/{dataset}/edge_lists_{fold}.pkl', 'rb'))
    edge_weight_data = pickle.load(open(f'data/{dataset}/edge_weights_{fold}.pkl', 'rb'))
    
    l0_array, l1_array = [], []
    save_path_l0 = f'saved_zz_gril/zz_graphs/{dataset}/{num_center_pts}_l0_fold_{fold}.npy'
    save_path_l1 = f'saved_zz_gril/zz_graphs/{dataset}/{num_center_pts}_l1_fold_{fold}.npy'
    
    for idx in range(len(edge_list_data)):
        # print('Starting to compute ZZ graph', flush=True)
        zz = zzMultipersGraph(num_center_pts=num_center_pts, num_graphs_in_seq=45, num_vertices=10)
        l0, l1 = zz.compute_zz_landscape(edge_list_data[idx], edge_weight_data[idx])
        print(f'Done: [{idx}/{len(edge_weight_data)}]', flush=True)
        l0_array.append(l0)
        l1_array.append(l1)
    
    l0_array = np.array(l0_array)
    l1_array = np.array(l1_array)
    np.save(save_path_l0, l0_array)
    np.save(save_path_l1, l1_array)

    
    