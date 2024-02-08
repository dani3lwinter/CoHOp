import os

os.environ["DGLBACKEND"] = "pytorch"

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=-1, dropout=0.2, device="cpu"):
        super(Model, self).__init__()
        self.is_linear = hidden_dim <= 0
        self.output_dim = output_dim

        if self.is_linear:
            self.W = nn.Linear(input_dim, output_dim)

        else:
            self.input_norm = nn.BatchNorm1d(input_dim)
            self.W1 = nn.Linear(input_dim, hidden_dim)
            self.W2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.BatchNorm1d(hidden_dim)
            self.activation = F.relu

        self.device = device
        self.to(device)


    def forward(self, x):
        if self.is_linear:
            logit = self.W(x)
        else:
            x = self.input_norm(x)
            x = self.W1(x)
            x = self.activation(x)
            x = self.dropout(self.norm1(x))
            logit = self.W2(x)

        return logit


    def predict_proba(self, x):
        x = x.to(self.device)
        if self.output_dim == 1:
            return F.sigmoid(self(x)).squeeze()
        else:
            return self(x).softmax(1)

    def predict(self, x):
        if self.output_dim == 1:
            return self.predict_proba(x)
        else:
            return self.predict_proba(x).max(1)