import pickle
import numpy as np
from model import zzMultipersGraph
from argparse import ArgumentParser
import os
from aeon.datasets import load_classification


def create_chunks(data, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    for start in range(0, data.shape[1] - chunk_size + 1, step):
        chunks.append(data[:, start:start + chunk_size])
    return chunks

def convert_to_numeric_labels(y):
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    y_numeric = y_numeric.astype(np.int32)
    return y_numeric

def compute_edge_list(corr_matrix):
    edge_list, edge_weight = [], []
    num_nodes = corr_matrix.shape[0]
    num = np.random.randint(25, 35)
    c = corr_matrix
    corr_matrix = np.nan_to_num(corr_matrix)
    corr_matrix = ((corr_matrix) + 1)/2
    threshold = np.percentile(np.abs(corr_matrix), num)  
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Upper triangle only to avoid duplicates
            weight = corr_matrix[i, j]
            if abs(weight) > threshold:  # Optional thresholding
                edge_list.append([i, j])  # (Node1, Node2)
                edge_weight.append(weight)
    if len(edge_weight) == 0:
        edge_list.append([0, 1])
        edge_weight.append(0.5)
    return edge_list, edge_weight

def process_time_series_to_edge_lists(data, chunk_size, overlap):
    chunks = create_chunks(data, chunk_size, overlap)
    edge_lists, edge_weights = [], []
    for chunk in chunks:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(chunk)
        # Convert to edge list
        edge_list, edge_weight = compute_edge_list(corr_matrix)
        edge_lists.append(edge_list)
        edge_weights.append(edge_weight)
    
    return edge_lists, edge_weights


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='NATOPS')
    args.add_argument('--num_center_pts', type=int, default=36)
    args.add_argument('--numvert', type=int, default=24)
    args = args.parse_args()
    
    dataset = args.dataset
    num_center_pts = args.num_center_pts
    
    print(f'Processing dataset: {dataset}')
    
    X_train, y_train = load_classification(dataset, split='train')
    X_test, y_test = load_classification(dataset, split='test')
    y_train = convert_to_numeric_labels(y_train)
    y_test = convert_to_numeric_labels(y_test)
    
    chunk_size = min(X_train.shape[2]//5, 128)
    overlap = int(0.7 * chunk_size)
    
    path = f'data/UCR_time_series/{args.dataset}/'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    train_edge_lists, train_edge_weights, train_labels = [], [], []
    for i in range(len(X_train)):
        data = X_train[i]
        label = y_train[i]
        
        # Process data
        
        edge_list, edge_weight = process_time_series_to_edge_lists(data, chunk_size=chunk_size, overlap=overlap)
        train_edge_lists.append(edge_list)
        train_edge_weights.append(edge_weight)
        train_labels.append(label)
        
        print(f'Processed {i}th sample out of {len(X_train)}', end='\r')

    test_edge_lists, test_edge_weights, test_labels = [], [], []
    for i in range(len(X_test)):
        
        data = X_test[i]
        label = y_test[i]
        
        # Process data
        
        edge_list, edge_weight = process_time_series_to_edge_lists(data, chunk_size=chunk_size, overlap=overlap)
        test_edge_lists.append(edge_list)
        test_edge_weights.append(edge_weight)
        test_labels.append(label)
        
        print(f'Processed {i}th sample out of {len(X_test)}', end='\r')
    
    path = f'saved_zz_gril/zz_graphs/UCR_time_series/{dataset}/'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    
    l0_array, l1_array = [], []
    save_path_l0_train = f'saved_zz_gril/zz_graphs/UCR_time_series/{dataset}/{num_center_pts}_l0_train.npy'
    save_path_l1_train = f'saved_zz_gril/zz_graphs/UCR_time_series/{dataset}/{num_center_pts}_l1_train.npy'
    save_path_l0_test = f'saved_zz_gril/zz_graphs/UCR_time_series/{dataset}/{num_center_pts}_l0_test.npy'
    save_path_l1_test = f'saved_zz_gril/zz_graphs/UCR_time_series/{dataset}/{num_center_pts}_l1_test.npy'
    
    num_seq_graphs = len(train_edge_lists[0])
    
    for idx in range(len(train_edge_lists)):
        zz = zzMultipersGraph(num_center_pts=num_center_pts, num_graphs_in_seq=num_seq_graphs, num_vertices=args.numvert)
        l0, l1 = zz.compute_zz_landscape(train_edge_lists[idx], train_edge_weights[idx])
        print(f'Done: [{idx}/{len(train_edge_weights)}]', flush=True, end='\r')
        l0_array.append(l0)
        l1_array.append(l1)
    
    l0_array = np.array(l0_array)
    l1_array = np.array(l1_array)
    np.save(save_path_l0_train, l0_array)
    np.save(save_path_l1_train, l1_array)
    
    l0_array, l1_array = [], []
    
    for idx in range(len(test_edge_lists)):
        zz = zzMultipersGraph(num_center_pts=num_center_pts, num_graphs_in_seq=num_seq_graphs, num_vertices=args.numvert)
        l0, l1 = zz.compute_zz_landscape(test_edge_lists[idx], test_edge_weights[idx])
        print(f'Done: [{idx}/{len(test_edge_weights)}]', flush=True, end='\r')
        l0_array.append(l0)
        l1_array.append(l1)
    
    l0_array = np.array(l0_array)
    l1_array = np.array(l1_array)
    np.save(save_path_l0_test, l0_array)
    np.save(save_path_l1_test, l1_array)
    
    print('Done')