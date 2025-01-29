import numpy as np
from tqdm import tqdm
import pickle

def create_chunks(data, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    for start in range(0, data.shape[1] - chunk_size + 1, step):
        chunks.append(data[:, start:start + chunk_size])
    return chunks


def compute_edge_list(corr_matrix):
    edge_list, edge_weight = [], []
    num_nodes = corr_matrix.shape[0]
    num = np.random.randint(25, 35)
    corr_matrix = ((corr_matrix) + 1)/2
    threshold = np.percentile(np.abs(corr_matrix), num)  
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Upper triangle only to avoid duplicates
            weight = corr_matrix[i, j]
            if abs(weight) > threshold:  # Optional thresholding
                edge_list.append([i, j])  # (Node1, Node2)
                edge_weight.append(weight)
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
    data = np.load('./rawdata/ISRUC_S3/ISRUC_S3.npz', allow_pickle=True)

    # Each fold corresponds to one subject's data (ISRUC-S3 dataset)
    for i in range(len(data['Fold_data'])):
        fold_data = data['Fold_data'][i]
        fold_label = data['Fold_label'][i]
        

        # Process data
        edge_lists, edge_weights, labels = [], [], []
        for sample in tqdm(range(len(fold_data))):
            edge_list, edge_weight = process_time_series_to_edge_lists(fold_data[sample], chunk_size=128, overlap=64)
            edge_lists.append(edge_list)
            edge_weights.append(edge_weight)
            labels.append(fold_label[sample])

        # Save edge lists
        with open(f'data/ISRUC_S3/edge_lists_{i}.pkl', 'wb') as f:
            pickle.dump(edge_lists, f)
        with open(f'data/ISRUC_S3/edge_weights_{i}.pkl', 'wb') as f:
            pickle.dump(edge_weights, f)
        with open(f'data/ISRUC_S3/labels_{i}.pkl', 'wb') as f:
            pickle.dump(labels, f)
        
        print(f'Saved {i}th fold')