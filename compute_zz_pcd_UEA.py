from aeon.datasets import load_classification
import numpy as np
from model import zzMultipersPointCloud
import os
from argparse import ArgumentParser


def preprocess(full_data):
    processed_data = []
    chunk_size = int(max(5, full_data.shape[-1] // 128))
    overlap = max(4, int(0.7 * chunk_size))
    step = chunk_size - overlap
    for j in range(full_data.shape[0]):
        chunks = []
        data = full_data[j]
        for start in range(0, data.shape[1] - chunk_size + 1, step):
            chunks.append(data[:, start:start + chunk_size])
        processed_data.append(np.array(chunks))
    return np.array(processed_data)
   

def convert_to_numeric_labels(y):
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])
    y_numeric = y_numeric.astype(np.int32)
    return y_numeric

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset', type=str, default='NATOPS')
    args.add_argument('--num_center_pts', type=int, default=36)
    args = args.parse_args()
    
    dataset = args.dataset
    X_train, y_train = load_classification(dataset, split='train')
    X_test, y_test = load_classification(dataset, split='test')
    y_train = convert_to_numeric_labels(y_train)
    y_test = convert_to_numeric_labels(y_test)
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    print(X_test.shape, y_test.shape)
    
    num_center_pts = 36
    
    save_dir_path = f'./saved_zz_gril/zz_point_clouds/UCR_time_series/{dataset}_{num_center_pts}_center_pts/'
    
    if os.path.exists(save_dir_path):
        print(f'Path {save_dir_path} already exists')
    else:
        os.makedirs(save_dir_path)
        print(f'Path {save_dir_path} created')
    
    save_path_l0_train = f'./saved_zz_gril/zz_point_clouds/UCR_time_series/{dataset}_{num_center_pts}_center_pts/processed_train_l0.npy'
    save_path_l1_train = f'./saved_zz_gril/zz_point_clouds/UCR_time_series/{dataset}_{num_center_pts}_center_pts/processed_train_l1.npy'
    save_path_l0_test = f'./saved_zz_gril/zz_point_clouds/UCR_time_series/{dataset}_{num_center_pts}_center_pts/processed_test_l0.npy'
    save_path_l1_test = f'./saved_zz_gril/zz_point_clouds/UCR_time_series/{dataset}_{num_center_pts}_center_pts/processed_test_l1.npy'
    
    l0_array = []
    l1_array = []
    
    for i in range(len(X_train)):
        zz = zzMultipersPointCloud(num_center_pts=num_center_pts, num_point_clouds_in_seq=X_train.shape[1], max_alpha_square=X_train[i].shape[-1])
        l0, l1 = zz.compute_zz_landscape(X_train[i])
        l0_array.append(l0)
        l1_array.append(l1)
        print(f'Done: [{i}/{len(X_train)}]', flush=True, end='\r')
    
    l0 = np.array(l0_array)
    l1 = np.array(l1_array)
    np.save(save_path_l0_train, l0)
    np.save(save_path_l1_train, l1)
    
    l0_array = []
    l1_array = []
    
    for i in range(len(X_test)):
        zz = zzMultipersPointCloud(num_center_pts=num_center_pts, num_point_clouds_in_seq=X_test.shape[1], max_alpha_square=X_test[i].shape[-1])
        l0, l1 = zz.compute_zz_landscape(X_test[i])
        l0_array.append(l0)
        l1_array.append(l1)
        print(f'Done: [{i}/{len(X_test)}]', flush=True, end='\r')
    
    l0 = np.array(l0_array)
    l1 = np.array(l1_array)
    np.save(save_path_l0_test, l0)
    np.save(save_path_l1_test, l1)
    
    
    
