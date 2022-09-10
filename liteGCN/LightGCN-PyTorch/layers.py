import torch
import torch.nn as nn


class MessagePassing(nn.Module):

    def __init__(self, ft_i, ft_o):
        super(MessagePassing, self).__init__()
        self.ft_i = ft_i
        self.ft_o = ft_o
        self.linear_w1 = nn.Linear(in_features=ft_i, out_features=ft_o)
        self.linear_w2 = nn.Linear(in_features=ft_i, out_features=ft_o)


    def forward(self, norm_adj, emb):
        """
        implementation of Eq.(7) of paper "Neural graph collaborative filtering"
        https://arxiv.org/pdf/1905.08108.pdf
        """
        # norm_adj: L = D^(-0.5)(A)D^(-0.5)
        # emb    : E_{l-1}
        # emb_agg: L .* E_{l-1}
        # emb_aug: (L+I) .* E_{l-1}
        # Eq. (7)
        #   (L+I) .* E_{l-1} .* W1 + L .* E_{l-1} * E_{l-1} .* W2
        #  = emb_aug .* W1 + emb_agg * emb .* W2
        emb_agg = torch.sparse.mm(norm_adj, emb)
        emb_aug = emb_agg + emb

        # out = nn.LeakyReLU(negative_slope=0.2)(
        #     self.linear_w1(emb_aug) + self.linear_w2(torch.mul(emb_agg, emb))
        # )
        # out = nn.LeakyReLU(negative_slope=0.2)(
        #     self.linear_w1(emb_aug)
        # )
        # out = nn.LeakyReLU(negative_slope=0.2)(emb_agg)
        out = emb_agg

        return out