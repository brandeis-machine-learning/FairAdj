import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


def grl_hook():
    def fun1(grad):
        return -1 * grad.clone()

    return fun1


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.sparse.mm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return self.dc(x), x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), z, mu, logvar

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1.}]


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size, dim_out, lr_mult):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, dim_out)
        self.relu = nn.ReLU()
        self.lr_mult = lr_mult

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = x * 1.0
        x.register_hook(grl_hook())
        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.ad_layer2(x)
        return x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": self.lr_mult}]
