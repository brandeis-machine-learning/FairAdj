# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import scipy.sparse as sp

import torch
from torch import optim

from args import parse_args
from utils import fix_seed
from dataloader import get_dataset
from model.gae import GCNModelVAE, AdversarialNetwork
from model.utils import preprocess_graph
from model.optimizer import loss_function, adv_loss_function
from eval import fair_link_eval


def main(args):
    G, adj, features, sensitive, test_edges_true, test_edges_false = get_dataset(args.dataset, args.scale,
                                                                                 args.test_ratio)

    n_nodes, feat_dim = features.shape
    features = torch.from_numpy(features).float().to(args.device)

    # Pre-processing
    adj_norm = preprocess_graph(adj).to(args.device)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray()).to(args.device)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = torch.Tensor([pos_weight]).to(args.device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    binary = True if max(sensitive) == 1 else False
    if binary:
        ad_net = AdversarialNetwork(args.hidden2, args.hidden2, 1, args.lr_mult).to(args.device)
    else:
        ad_net = AdversarialNetwork(args.hidden2, args.hidden2, max(sensitive) + 1, args.lr_mult).to(args.device)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout).to(args.device)
    optimizer = optim.Adam(model.get_parameters() + ad_net.get_parameters(), lr=args.lr)

    adv_target = torch.from_numpy(sensitive).float().to(args.device)

    model.train()
    ad_net.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        recovered, z, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)

        adv_loss = adv_loss_function(z, adv_target, ad_net, binary)
        total_loss = loss + args.alpha * adv_loss
        total_loss.backward()
        optimizer.step()

        loss = loss.item()
        adv_loss = adv_loss.item()

        print("Epoch: [{:d}/{:d}];".format((epoch + 1), args.epochs),
              "Loss: {:.3f};".format(loss),
              "Adv. loss: {:.3f};".format(adv_loss),
              )

    model.eval()
    with torch.no_grad():
        z = model(features, adj_norm)[1]
    hidden_emb = z.data.cpu().numpy()

    std = fair_link_eval(hidden_emb, sensitive, test_edges_true, test_edges_false)
    col = ["auc", "ap", "dp", "true", "false", "fnr", "tnr"]
    print("Result below ------")
    for term, val in zip(col, std):
        print(term, ":", val)

    return


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device(args.device)
    fix_seed(args.seed)
    main(args)
