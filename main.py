import dgl
import numpy as np
import torch
import argparse

from utils import (set_seed, validate_device, build_label_histograms, fast_build_label_histograms,
                   graph_split,)
from trainer import Trainer, minibatch_accuracy
from dataloader import  load_data
from model import Model
import yaml

OGBN_DATASETS = ['ogbn-arxiv', 'ogbn-products']

def run_once(dataset, args, seed=0, config=None):
    set_seed(seed)
    device = validate_device(args.gpu)
    use_histograms = not args.dont_augment

    # load config file into dict
    if config is None:
        with open('hyper_parameters.yaml') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
            config = {
                **config["defaults"],
                **config[dataset]
            }
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.n_iterations is not None:
            config['n_iterations'] = args.n_iterations

    # load dataset and split indices
    g, labels, idx_train, idx_val, idx_test = load_data(
        dataset, "data", seed=seed,
        labelrate_train=args.train_labelrate,
        labelrate_val=args.val_labelrate
    )
    g = dgl.remove_self_loop(g)
    x = g.ndata["feat"].float()
    y = labels
    num_classes = y.int().max().item() + 1
    if num_classes == 2:
        num_classes = 1

    # split the test further into seen and unseen set of nodes
    if args.unseen_rate > 0:
        # inductive setting
        seen_idx_train, seen_idx_val, seen_idx_test, idx_seen, unseen_idx_test = (
            graph_split(idx_train, idx_val, idx_test, args.unseen_rate, seed))
        seen_g, seen_x, seen_y = g.subgraph(idx_seen), x[idx_seen], y[idx_seen]
        if args.verbose > 1:
            print(f"train/val/seen-test/unseen-test: {len(idx_train)}/{len(idx_val)}/"
                  f"{len(seen_idx_test)}/{len(unseen_idx_test)}")
    else:
        # transductive setting
        seen_idx_train, seen_idx_val, seen_idx_test = idx_train, idx_val, idx_test
        unseen_idx_test = None
        idx_seen = torch.arange(len(labels))
        seen_g, seen_x, seen_y = g, x, y
        if args.verbose > 1:
            print(f"train/val/test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")

    histogram_builder = fast_build_label_histograms if dataset in OGBN_DATASETS \
        else build_label_histograms
    # build neighbor histogram of the train labels
    if use_histograms:
        neighbor_hist = histogram_builder(seen_g, seen_idx_train, seen_y[seen_idx_train],
                                          num_classes,
                                          alpha=config["smoothing_radius"],
                                          cutoff=config["histogram_depth"],
                                          batch_size=config["batch_size"])
        seen_x = torch.cat([seen_x, neighbor_hist.float()], dim=1)
    # return (accuracy_evaluator(neighbor_hist[seen_idx_test], seen_y[seen_idx_test]),)
    # initialize the model and train it
    model = Model(input_dim=seen_x.shape[1], output_dim=num_classes, device=device,
                  hidden_dim=config["hidden_dim"],
                  dropout=config['dropout_ratio'] if 'dropout_ratio' in config else None)
    # print(model)
    trainer = Trainer(config, device)
    trainer.iterative_training(model, seen_g, num_classes, seen_x,
                               seen_idx_train, seen_idx_val,
                               seen_y[seen_idx_train], seen_y[seen_idx_val],
                               verbose=args.verbose)

    # evaluate on the regular seen test
    score_seen = minibatch_accuracy(model, seen_x[seen_idx_test].to(device),
                               seen_y[seen_idx_test].to(device), config["batch_size"])
    if unseen_idx_test is None:
        return (score_seen,)

    # evaluate on the inductive test set
    if use_histograms:
        neighbor_hist = histogram_builder(g, idx_seen[seen_idx_train], seen_y[seen_idx_train],
                                          num_classes,
                                          alpha=config["smoothing_radius"],
                                          cutoff=config["histogram_depth"],
                                          batch_size=config["batch_size"])
        x = torch.cat([x, neighbor_hist.float()], dim=1)
    score_unseen = minibatch_accuracy(model, x[unseen_idx_test].to(device),
                                    y[unseen_idx_test].to(device), config["batch_size"])

    return score_seen, score_unseen


def main(args):

    for dataset in args.datasets:
        print(f"==== {dataset} ====")

        seen_accuracies, unseen_accuracies  = [], []
        for i in range(args.n_repeats):
            scores = run_once(dataset, args, seed=i)
            seen_accuracies.append(scores[0])
            if args.verbose > 0:
                print(f"seen-test accuracy: {scores[0] * 100:.2f}%", end='')

            if len(scores) > 1:
                unseen_accuracies.append(scores[1])
                if args.verbose > 0:
                    print(f", unseen-test accuracy: {scores[1] * 100:.2f}%", end='')

            if args.verbose > 0:
                print(f' (seed {i})')

        seen_accuracies = np.array(seen_accuracies) * 100
        print(f"{dataset:<10} - seen {seen_accuracies.mean():.2f}% ± {seen_accuracies.std():.2f}%")

        if len(unseen_accuracies) > 0:
            unseen_accuracies = np.array(unseen_accuracies) * 100
            print(f"{dataset:<10} - inductive test: {unseen_accuracies.mean():.2f}% "
                  f"± {unseen_accuracies.std():.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*', type=str,
                        help="The datasets to run the model on.\n"
                             "Supported datasets: cora, citeseer, pubmed, a-computer, a-photo, "
                             "ogbn-arxiv, ogbn-products.\n")

    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. If -1, will use CPU. Default: 0")

    parser.add_argument("--verbose", type=int, default=1,
                        help="Logging level - 0 indicates minimal printing, and a value of 3"
                             "results in the printing of all available information. Default: 1")

    parser.add_argument("--n_repeats", type=int, default=10,
                        help="Number of times to run with different seeds. Default: 10")

    parser.add_argument("--train_labelrate", type=float, default=20,
                        help="The portion of samples used as training set. If its larger than 1, "
                             "it will be used as number of instances per class. Default: 20")

    parser.add_argument("--val_labelrate", type=float, default=30,
                        help="The portion of samples used as validation set. If its larger than 1,"
                             " it will be used as number of instances per class. Default: 30")

    parser.add_argument("--unseen_rate", type=float, default=0,
                        help="Portion of the test-set that will be held out as unseen test "
                             "(i.e. inductive test). 0 means only running in transductive settings"
                             " Default: 0")

    parser.add_argument("--dont_augment", default=False, action='store_true',
                        help="If set, do not augment the input features by concatenating "
                             "histograms.")

    parser.add_argument("--gamma", type=float, default=None,
                        help="The weight of the consistency loss.")

    parser.add_argument("--n_iterations", type=int, default=None,
                        help="Number of iterations to train on pseudo-labels.")

    args = parser.parse_args()
    if len(args.datasets) == 0:
        parser.error('At least one dataset name is required.')

    main(args)