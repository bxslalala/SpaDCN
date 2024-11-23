import torch
import random
import time
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
import sys
import os


def dense_tensor_to_sparse(x):
    """ Converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x, as_tuple=False)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def get_edge_inf_batch(eadj, Tmatrix, edge_feature, sampled_node):
    Tmatrix = Tmatrix[sampled_node, :]
    Tmatrix_col_sum = Tmatrix.sum(0)
    edge_index = (Tmatrix_col_sum == 2).nonzero().view(-1)
    return dense_tensor_to_sparse(eadj[edge_index, :][:, edge_index]), dense_tensor_to_sparse(Tmatrix[:, edge_index]), edge_feature[edge_index, :]


def minibatch_generator(inputs, batchsize):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    num_train = len(inputs[6])
    n_val_test = len(inputs[7]) + len(inputs[8])
    n_batch = n_val_test // (batchsize - num_train)
    
    n_val_batch = len(inputs[7]) // n_batch
    n_test_batch = len(inputs[8]) // n_batch

    data_list = []

    tmp_dense_adj = inputs[0].to_dense()
    tmp_dense_eadj = inputs[3].to_dense()
    tmp_dense_tmatrix = inputs[4].to_dense()
    idx_val_shuffle = inputs[7][torch.randperm(len(inputs[7]))]
    idx_test_shuffle = inputs[8][torch.randperm(len(inputs[8]))]

    for i in range(n_batch):
        idx_val_batch = idx_val_shuffle[range(i * n_val_batch, (i+1) * n_val_batch)]
        idx_test_batch = idx_test_shuffle[range(i * n_test_batch, (i+1) * n_test_batch)]
        idx_batch = torch.cat([inputs[6], idx_val_batch, idx_test_batch])
        eadj_batch, T_batch, efeature_batch = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], idx_batch)
        data_list.append([dense_tensor_to_sparse(tmp_dense_adj[idx_batch][:, idx_batch]), 
                         inputs[1][idx_batch],
                         inputs[2][idx_batch],
                         eadj_batch,
                         T_batch,
                         efeature_batch,
                         torch.tensor(range(num_train)),
                         torch.tensor(range(num_train, num_train + n_val_batch))])
    return data_list


def minibatch_generator_test(inputs, batchsize):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    num_test = len(inputs[6])
    n_batch = num_test // batchsize

    data_list = []
    # Pre-convert the sparse 
    tmp_dense_adj = inputs[0].to_dense()
    tmp_dense_eadj = inputs[3].to_dense()
    tmp_dense_tmatrix = inputs[4].to_dense() 

    for i in range(n_batch - 1):
        idx_batch = inputs[6][range(i * batchsize, (i + 1) * batchsize)]
        eadj_batch, T_batch, efeature_batch = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], idx_batch)
        data_list.append([dense_tensor_to_sparse(tmp_dense_adj[idx_batch][:, idx_batch]),
                         inputs[1][idx_batch],
                         inputs[2][idx_batch],
                         eadj_batch,
                         T_batch,
                         efeature_batch])
    # Handle remaining nodes
    reminder_n = num_test - (n_batch - 1) * batchsize
    reminder_idx = inputs[6][range((n_batch - 1) * batchsize, num_test)]
    eadj_reminder, T_reminder, efeature_reminder = get_edge_inf_batch(tmp_dense_eadj, tmp_dense_tmatrix, inputs[5], reminder_idx)
    data_list.append([dense_tensor_to_sparse(tmp_dense_adj[reminder_idx][:, reminder_idx]),
                      inputs[1][reminder_idx],
                      inputs[2][reminder_idx],
                      eadj_reminder,
                      T_reminder,
                      efeature_reminder])
    return data_list


def create_direction_feature(directed_adj, edge_names):
    '''Return a numpy array, each row represents an edge, with two dimensions representing direction.'''
    direction_feat = np.zeros((len(edge_names), 2), dtype=int)
    for i, x in enumerate(edge_names):
        if directed_adj[x[0], x[1]] == 1:
            direction_feat[i, :] = [1, 0]
        else:
            direction_feat[i, :] = [0, 1]
    return direction_feat


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_mse(output, labels):
    diff = output - labels
    return torch.sum(diff * diff) / diff.numel()


def RMSELoss(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    with open(filename) as file:
        for line in file:
            index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)


def encode_onehot(labels):
    classes = sorted(set(labels))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array([classes_dict[label] for label in labels], dtype=np.int32)
    return labels_onehot


def create_edge_adj_new(conns):
    '''
    Create an edge adjacency matrix from vertex adjacency matrix.
    '''
    vertex_adj = conns.copy()
    np.fill_diagonal(vertex_adj, 0)
    edge_index = np.nonzero(np.triu(vertex_adj))
    num_edge = len(edge_index[0])
    edge_name = list(zip(edge_index[0], edge_index[1]))

    edge_adj = np.zeros((num_edge, num_edge))
    for i in range(num_edge):
        for j in range(i, num_edge):
            if set(edge_name[i]) & set(edge_name[j]):
                edge_adj[i, j] = 1
    adj = edge_adj + edge_adj.T
    np.fill_diagonal(adj, 1)
    return sp.csr_matrix(adj), edge_name


def node_corr_cosine(adj, feat):
    """Calculate edge cosine distance."""
    distance = squareform(pdist(feat, 'cosine'))
    edge_feat = distance[np.nonzero(sp.triu(adj, k=1))]
    ret = edge_feat.reshape((len(edge_feat), 1))
    return ret


def create_transition_matrix_new(conns):
    '''Create N_v * N_e transition matrix.'''
    vertex_adj = conns.copy()
    np.fill_diagonal(vertex_adj, 0)
    edge_index = np.nonzero(np.triu(vertex_adj))
    num_edge = len(edge_index[0])
    edge_name = list(zip(edge_index[0], edge_index[1]))

    row_index = [i for sub in edge_name for i in sub]
    col_index = np.repeat(range(num_edge), 2)
    data = np.ones(num_edge * 2)
    T = sp.csr_matrix((data, (row_index, col_index)), shape=(vertex_adj.shape[0], num_edge))

    return T


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1)).astype(float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
