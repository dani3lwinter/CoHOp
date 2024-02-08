import networkx as nx
import scipy.sparse as sparse
import random
import numpy as np
import os

os.environ["DGLBACKEND"] = "pytorch"

from tqdm import tqdm
import dgl
import torch
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
import warnings
from dgl.dataloading import DataLoader
from torcheval import metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def idx_split(idx, ratio, seed=0):
    """
    randomly split idx into two portions with ratio% elements and (1 - ratio)% elements
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    # assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2

def graph_split(idx_train, idx_val, idx_test, rate, seed):
    """
    Args:
        The original setting was transductive. Full graph is observed, and idx_train takes up a small portion.
        Split the graph by further divide idx_test into [idx_test_tran, idx_test_ind].
        rate = idx_test_ind : idx_test (how much test to hide for the inductive evaluation)

        Ex. Ogbn-products
        loaded     : train : val : test = 8 : 2 : 90, rate = 0.2
        after split: train : val : test_tran : test_ind = 8 : 2 : 72 : 18

    Return:
        Indices start with 'obs_' correspond to the node indices within the observed subgraph,
        where as indices start directly with 'idx_' correspond to the node indices in the original graph
    """
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1: N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2:]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def validate_device(gpu_id):
    """Validate the input GPU ID is valid on the given environment.
    If no GPU is presented, return 'cpu'.
    Parameters
    ----------
    gpu_id : int
        GPU ID to check.
    Returns
    -------
    device : str
        Valid device, e.g., 'cuda:0' or 'cpu'.
    """

    # cast to int for checking
    gpu_id = int(gpu_id)

    # if it is cpu
    if gpu_id == -1:
        return 'cpu'

    # if gpu is available
    if torch.cuda.is_available() and 0 <= gpu_id and gpu_id < torch.cuda.device_count():
        device = 'cuda:{}'.format(gpu_id)
    else:
        warnings.warn(f'The cuda:{gpu_id} is not available. Set to cpu.')
        device = 'cpu'

    return device

def batch_conv(batch_size, conv_func):
    def batch_conv_func(g, x):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        res = []
        dataloader = DataLoader(g, torch.arange(x.shape[0], device=g.device), sampler,
                                device=g.device,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=0,
                                )

        # with dataloader.enable_cpu_affinity():
        for step, (src_nodes, dst_nodes, mfgs) in enumerate(dataloader):
            res.append(conv_func(mfgs[0], (x[src_nodes], x[dst_nodes])))

        return torch.concatenate(res)

    return batch_conv_func




def build_label_histograms_old(g, idx_train, y_train, num_classes, alpha, cutoff=None, batch_size=None):
    """
    implementation with low ram usage
    """
    g_nx = dgl.to_networkx(g)
    y_train = y_train.numpy()

    histograms = np.zeros((g.number_of_nodes(), num_classes))

    for i, label in enumerate(y_train):
        index_in_graph = idx_train[i].item()
        result = np.array(list(nx.single_target_shortest_path_length(g_nx, index_in_graph, cutoff=cutoff)))
        sources, distances = result[:, 0], result[:, 1]
        histograms[sources, label] += alpha ** distances

    histograms /= (histograms.sum(1) + 1e-6)[:, None]

    return torch.tensor(histograms)

def build_label_histograms(g, idx_train, y_train, num_classes, alpha, cutoff=None, batch_size=None,
                           out_subset=None):
    dist_mat = sparse.csgraph.dijkstra(
            g.adj_external(scipy_fmt="csr"),
            directed=False,
            indices=idx_train if out_subset is None else out_subset,
            unweighted=True,
            limit=cutoff
        )
    dist_mat = torch.tensor(dist_mat)
    W = torch.where(dist_mat <= cutoff, alpha**dist_mat, 0).float()

    if out_subset is None:
        W = W.T # Output histogram for all the nodes
    else:
        W = W[:, idx_train]  # output only the histogram of nodes in out_subset

    H = W.to(y_train.device) @ to_one_hot(y_train, num_classes=num_classes)

    return H / (H.sum(1) + 1e-6)[:, None]



def fast_build_label_histograms(g, idx_train, y_train, num_classes, alpha, cutoff, batch_size=-1,
                                out_subset=None):
    conv = GraphConv(0, 0, norm='right', weight=False, bias=False,
                     allow_zero_in_degree=True)

    if batch_size > 0:
        conv = batch_conv(batch_size, conv)

    histograms = torch.zeros((g.number_of_nodes(), num_classes), device=y_train.device).float()
    histograms[idx_train] = to_one_hot(y_train, num_classes)

    histograms = conv(g, alpha * histograms)
    for i in range(1, cutoff-1):
        histograms += conv(g, alpha * histograms)

    histograms = histograms / (histograms.sum(1) + 1e-6)[:, None]
    if out_subset is None:
        return histograms
    return histograms[out_subset]

def accuracy(y_pred, y_true):
    if len(y_true.shape) > 1:
        y_true = y_true.argmax(1)
    if len(y_pred.shape) == 1:
        return metrics.functional.binary_auroc(y_pred, y_true).item()
    else:
        pred = y_pred.argmax(1)
        return pred.eq(y_true).float().mean().item()


def to_one_hot(classes_vec, num_classes):
    if num_classes == 1:
        return classes_vec.unsqueeze(1).float()
    else:
        return F.one_hot(classes_vec, num_classes=num_classes).float()

