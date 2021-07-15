import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KLD


def adv_loss_function(emb: torch.FloatTensor, sensitive: torch.Tensor, ad_net: torch.nn.Module, binary: bool = True,
                      pos_weight: torch.FloatTensor = None) -> torch.Tensor:
    ad_out = ad_net(emb)
    if binary:
        sensitive = torch.reshape(sensitive, (sensitive.shape[0], 1))
        if pos_weight:
            return F.binary_cross_entropy_with_logits(ad_out, sensitive, pos_weight=pos_weight)
        else:
            return F.binary_cross_entropy_with_logits(ad_out, sensitive)
    else:
        sensitive = sensitive.long()
        return F.cross_entropy(ad_out, sensitive)
