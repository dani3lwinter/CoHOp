import numpy as np

from dataloader import load_data
import dgl
import time
from utils import fast_build_label_histograms, build_label_histograms
datasets = ["cora", "citeseer", "pubmed", "a-computer", "a-photo"]

N_REPEATS = 10

def measure_time(dataset, method, cutoff):
    times = []
    for i in range(N_REPEATS):
        g, labels, idx_train, idx_val, idx_test = load_data(
            dataset,
            "../data",
            split_idx=0,
            seed=i,
            labelrate_train=20,
            labelrate_val=30,
        )
        g = dgl.remove_self_loop(g)
        y = labels
        num_classes = y.int().max().item() + 1

        start_time = time.time()
        neighbor_hist = method(g, idx_train, y[idx_train], num_classes, alpha=0.5, cutoff=cutoff)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    return np.array(times).mean()

if __name__ == '__main__':
    for cutoff in [2, 4, 6, 8, 10]:
        print(f"cutoff = {cutoff}")
        print(f" Dataset  | exact | aprox (milisec)")

        for dataset in datasets:
            t1 = measure_time(dataset, build_label_histograms, cutoff)
            t2 = measure_time(dataset, fast_build_label_histograms, cutoff)
            print(f"{dataset:<10}| {int(t1):^5} | {int(t2):^5}")
        print()
