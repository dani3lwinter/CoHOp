
import os
import copy

import numpy as np
from utils import to_one_hot

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torcheval import metrics
from dgl.dataloading import DataLoader

def minibatch_epoch(x, y, model, optimizer, all_dataloader, train_dataloader, gamma):
    consistency_conv = EntropyConv()
    accumulated_loss = 0
    train_iter = iter(train_dataloader)
    n_batches = len(all_dataloader)

    pbar = tqdm(all_dataloader)
    for src_nodes, dst_nodes, mfgs in pbar:

        src_preds = model(x[src_nodes]).softmax(1)
        dst_preds = model(x[dst_nodes]).softmax(1)

        src_preds = torch.where(src_preds < 1e-6, 1e-6, src_preds)
        dst_preds = torch.where(dst_preds < 1e-6, 1e-6, dst_preds)
        consistency_loss = consistency_conv(mfgs[0], (src_preds, dst_preds)).mean()

        next_gt_nodes = next(train_iter, None)
        if next_gt_nodes is None:
            train_iter = iter(train_dataloader)
            next_gt_nodes = next(train_iter, None)
        gt_nodes = next_gt_nodes[1]

        gt_loss = F.cross_entropy(model(x[gt_nodes]), y[gt_nodes])
        loss = gt_loss + gamma * consistency_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            f"loss: {loss.item():.4f}")

        accumulated_loss += loss.item()
    return accumulated_loss / n_batches

@torch.no_grad()
def minibatch_eval(model, x, batch_size):
    if batch_size <= 0:
        return model.predict_proba(x)

    predictions = []

    all_indices = torch.arange(x.shape[0])
    for i in range(0, x.shape[0], batch_size):
        batch_indices = all_indices[i:i + batch_size]
        batch_x = x[batch_indices]
        predictions.append(model.predict_proba(batch_x))

    return torch.concatenate(predictions)

@torch.no_grad()
def minibatch_accuracy(model, x, y, batch_size):
    if batch_size <= 0:
        return accuracy_evaluator(model.predict_proba(x), y)

    total_hits = 0
    all_indices = torch.randperm(x.shape[0])

    for i in range(0, x.shape[0], batch_size):
        batch_indices = all_indices[i:i + batch_size]
        batch_x, batch_y = x[batch_indices], y[batch_indices]
        pred = model(batch_x).argmax(1)
        total_hits += pred.eq(batch_y).sum().item()

    return float(total_hits) / x.shape[0]

def accuracy_evaluator(out, labels):
    if len(out.shape) == 1:
        return metrics.functional.binary_auroc(out, labels).item()
    else:
        if len(labels.shape) > 1:
            labels = labels.argmax(1)
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

class EntropyConv(nn.Module):
    def __init__(self):
        super(EntropyConv, self).__init__()


    def entropy_message_func(self, edges):  # gamma=0.00005
        src = edges.src["h"]
        dst = edges.dst["h"]
        m = - (torch.log(src) * dst).sum(1)
        return {"m": m}

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            if g.is_block:
                g.srcdata['h'], g.dstdata['h'] = h
            else:
                g.ndata['h'] = h

            g.update_all(self.entropy_message_func,fn.mean("m", "h_N"))
            return g.dstdata['h_N'] if g.is_block else g.ndata['h_N']



class Trainer:

    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.gamma = config["gamma"]

        self.n_epochs = config["n_epochs"]
        self.patience = config["patience"]
        self.batch_size = config["batch_size"]

        self.n_iterations = config["n_iterations"]
        self.threshold = config["threshold"]


    def train_single(self,
              model,
              g: dgl.DGLGraph,
              x, train_idx, val_idx, y_train, y_val,
              verbose=False):

        x = x.to(self.device)
        g = g.to(self.device)

        train_idx = train_idx.to(self.device)
        val_idx = val_idx.to(self.device)
        y_train = y_train.to(self.device)
        y_val = y_val.to(self.device)
        y = torch.zeros(x.shape[0], device=self.device, dtype=torch.long)
        y[train_idx] = y_train

        train_dataloader, val_dataloader = None, None
        if self.batch_size > 0:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

            train_dataloader = DataLoader(g, train_idx, sampler,
                device=self.device,
                batch_size=self.batch_size,
                shuffle=True,  # Whether to shuffle the nodes for every epoch
                drop_last=False,  # Whether to drop the last incomplete batch
                num_workers=0,  # Number of sampler processes
            )

            all_dataloader = DataLoader(g, torch.arange(len(g.nodes()), device=self.device),
                                        sampler,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        drop_last=False,
                                        num_workers=0,
                                        device=self.device,
            )

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if model.output_dim == 1:
            criterion = torch.nn.BCELoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        entropy_conv = EntropyConv()

        best_score_val, count = -1, 0

        show_progress = verbose and self.batch_size <= 0
        pbar = tqdm(range(1, self.n_epochs + 1)) if show_progress else range(1, self.n_epochs + 1)
        for epoch in pbar:
            model.train()
            if self.batch_size > 0 :
                loss = minibatch_epoch(x, y, model, optimizer, all_dataloader, train_dataloader, self.gamma)
            else:
                pred = model(x)
                gt_loss = criterion(pred[train_idx], y_train)

                crossentropy_loss = 0
                if self.gamma != 0 :
                    crossentropy_loss = entropy_conv(g, pred.softmax(1)).mean()

                loss = gt_loss + self.gamma * crossentropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # use validation set for early stopping
            model.eval()

            score_val = minibatch_accuracy(model, x[val_idx], y_val, self.batch_size)
            score_train = minibatch_accuracy(model, x[train_idx], y_train, self.batch_size)

            if score_val >= best_score_val :
                best_score_val = score_val
                state = copy.deepcopy(model.state_dict())
                count = 0
            else:
                count += 1

            if count == self.patience:
                break

            if show_progress:
                pbar.set_description(f"loss: {loss:.4f} | acc_train: {score_train:.4f} | acc_val: {score_val:.4f} ")


        model.load_state_dict(state)
        model.eval()
        return model

    def smooth_pseudo_labels(self, g, p_preds, idx_val, y_val):
        conv = GraphConv(0, 0, norm='right', weight=False, bias=False,
                         allow_zero_in_degree=True)
        smoothed_preds = conv(g, p_preds)
        # smoothed_preds /= (smoothed_preds.sum(1) + 1e-6)[:, None]

        best_val_acc, best_center_weight = -1, 1
        for center_weight in np.arange(0, 1.01, 0.01):
            new_preds = center_weight * p_preds + (1 - center_weight) * smoothed_preds
            val_acc = accuracy_evaluator(new_preds[idx_val], y_val)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_center_weight = center_weight

        return best_center_weight * p_preds + (1 - best_center_weight) * smoothed_preds
    def iterative_training(self,
              model,
              g: dgl.DGLGraph, num_classes,
              x, idx_train, idx_val, y_train, y_val,
              verbose=False):

        original_train_mask = torch.zeros(x.shape[0]).to(bool)
        original_train_mask[idx_train] = True

        num_nodes = x.shape[0]
        num_train_nodes = original_train_mask.sum().item()

        pseudo_labels = torch.zeros((num_nodes, )).to(torch.long)
        pseudo_labels[idx_train] = y_train
        high_conf_idx = idx_train.clone().detach()

        best_val_score, best_iter = -1, 0
        for i in range(self.n_iterations):

            self.train_single(model, g, x,
                              high_conf_idx, idx_val,
                              pseudo_labels[high_conf_idx], y_val,
                              verbose=verbose > 2)

            p_preds = minibatch_eval(model, x, self.batch_size).cpu()

            train_acc = accuracy_evaluator(p_preds[idx_train], y_train)
            val_acc = accuracy_evaluator(p_preds[idx_val], y_val)

            p_preds[idx_train] = to_one_hot(y_train, num_classes)
            smoothed_preds = self.smooth_pseudo_labels(g, p_preds, idx_val, y_val)

            # update pseudo labels
            pseudo_labels = smoothed_preds.argmax(1)
            pseudo_labels[idx_train] = y_train
            high_conf_mask = (smoothed_preds.max(1)[0] > self.threshold) | original_train_mask
            high_conf_idx = torch.where(high_conf_mask)[0]

            if val_acc > best_val_score:
                best_val_score, best_iter = val_acc, i
                state = copy.deepcopy(model.state_dict())

            if verbose > 1:
                print(f"[Iteration {i}] "
                      f"Val: {val_acc * 100:.1f}%   "
                      f"Train: {train_acc * 100:.1f}%  "
                      f"({num_train_nodes / num_nodes * 100:.1f}% trainset size)")

            num_train_nodes = high_conf_mask.sum().item()

        model.load_state_dict(state)

