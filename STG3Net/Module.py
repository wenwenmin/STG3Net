import copy
import numpy as np
import random
import torch.nn.functional as F
import torch
from torch import nn
from torch_geometric.nn import (
    TransformerConv,
    LayerNorm,
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

import faiss

def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def create_activation(name=None):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "iden":
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "lrelu":
        return nn.LeakyReLU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def full_block(in_features, out_features, p_drop, act=nn.ELU()):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        act,  # nn.ELU(),
        nn.Dropout(p=p_drop),
    )

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2, act=F.relu, bn=True, graphtype="gcn"):
        super(GraphConv, self).__init__()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.in_features = in_features
        self.out_features = out_features
        self.bn = bn(out_features)
        self.act = act
        self.dropout = dropout
        if graphtype == "gcn":
            self.conv = GCNConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gat": # Default heads=1
            self.conv = GATConv(in_channels=self.in_features, out_channels=self.out_features)
        elif graphtype == "gin": # Default heads=1
            self.conv = TransformerConv(in_channels=self.in_features, out_channels=self.out_features)
        else:
            raise NotImplementedError(f"{graphtype} is not implemented.")

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = config['feat_hidden1']
        self.feat_hidden2 = config['feat_hidden2']
        self.gcn_hidden = config['gcn_hidden']
        self.latent_dim = config['latent_dim']

        self.p_drop = config['p_drop']
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))
        # GCN layers
        self.gc1 = GraphConv(self.feat_hidden2, self.gcn_hidden, dropout=self.p_drop, act=F.relu)
        self.gc2 = GraphConv(self.gcn_hidden, self.latent_dim, dropout=self.p_drop, act=lambda x: x)

    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = self.gc1(x, edge_index)
        x = self.gc2(x, edge_index)
        return x

class Decoder(nn.Module):
    def __init__(self, output_dim, config):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = config['latent_dim']
        self.p_drop = config['p_drop']
        self.layer1 = GraphConv(self.input_dim, self.output_dim, dropout=self.p_drop, act=nn.Identity())

    def forward(self, x, edge_index):
        return self.layer1(x, edge_index)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.dec_in_dim = config['latent_dim']
        self.encoder = Encoder(input_dim, config)

        self.encoder_to_decoder = nn.Linear(self.dec_in_dim, self.dec_in_dim, bias=False)
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        self.decoder = Decoder(input_dim, config)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        # self.rep_mask = nn.Parameter(torch.zeros(1, self.dec_in_dim))
        self.replace_rate = 0.1
        self.mask_token_rate = 1 - self.replace_rate
        self.mask_rate = config['mask_rate']
        self.t = config['t']
        self.anchor_pair = None
        self.mask_method = config['mask_method']

    def encoding_mask_noise(self, x, edge_index, mask_rate=0.3, method='spot'):
        if method == 'gene':
            return self._mask_by_genes(x, edge_index, mask_rate)
        elif method == 'spot':
            return self._mask_by_nodes_(x, edge_index, mask_rate)
        else:
            raise Exception

    def _mask_by_nodes_(self, x, edge_index, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self.replace_rate > 0:
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_edge_index = edge_index.clone()
        return out_x, use_edge_index, mask_nodes

    def _mask_by_genes(self, x, edge_index, mask_rate=0.3):
        use_x = x.clone()
        if isinstance(use_x, np.ndarray):
            mask = np.random.choice([True, False], size=use_x.shape, p=[mask_rate, 1 - mask_rate], replace=False)
        elif isinstance(use_x, torch.Tensor):
            mask = torch.rand(use_x.shape) < mask_rate
        else:
            raise TypeError("type error!")
        mask = mask.to(use_x.device)
        use_x[mask] = 0
        use_edge_index = edge_index.clone()
        return use_x, use_edge_index, mask


    def set_anchor_pair(self, anchor_pair):
        self.anchor_pair = anchor_pair

    def mask_attr_prediction(self, x, edge_index):
        use_x, use_adj, mask = self.encoding_mask_noise(x, edge_index, self.mask_rate, method=self.mask_method)
        enc_rep = self.encoder(use_x, use_adj)

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)
        recon = self.decoder(rep, use_adj)

        x_init = x[mask]
        x_rec = recon[mask]

        if self.anchor_pair is not None:
            (anchor, positive, negative) = self.anchor_pair
            tri_loss = self.triplet_loss(enc_rep, anchor, positive, negative)
        else:
            tri_loss = 0

        rec_loss = self.sce_loss(x_rec, x_init, t=self.t)

        return tri_loss, rec_loss, enc_rep


    def sce_loss(self, x, y, t=2):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
        loss = -torch.log(cos_m.pow_(t))
        return loss.mean()

    def triplet_loss(self, emb, anchor, positive, negative, margin=1.0):
        anchor_arr = emb[anchor]
        positive_arr = emb[positive]
        negative_arr = emb[negative]
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_output = triplet_loss(anchor_arr, positive_arr, negative_arr)
        return tri_output

    def forward(self, x, edge_index):
        return self.mask_attr_prediction(x, edge_index)

    @torch.no_grad()
    def embeding(self, x, edge_index):
        use_x, use_adj, mask = self.encoding_mask_noise(x, edge_index, self.mask_rate, method=self.mask_method)
        enc_rep = self.encoder(use_x, use_adj)
        return enc_rep

    @torch.no_grad()
    def evaluate(self, x, edge_index):
        enc_rep = self.encoder(x, edge_index)
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(rep, edge_index)
        return enc_rep, recon



class Discriminator(nn.Module):
    def __init__(self, class_num, label, config):
        super(Discriminator, self).__init__()
        self.class_num = class_num
        self.label = label
        self.latent_dim = config['latent_dim']
        self.dic_hidden = config['dic_hidden']
        self.p_drop = config['p_drop']
        # discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('discriminator_L1', full_block(self.latent_dim, self.dic_hidden, self.p_drop, act=create_activation("lrelu")))
        self.discriminator.add_module('discriminator_L2', full_block(self.dic_hidden, self.dic_hidden, self.p_drop, act=create_activation("lrelu")))
        self.discriminator.add_module('discriminator_L3', nn.Linear(self.dic_hidden, self.class_num))

    @torch.no_grad()
    def evaluate(self, x):
        x = self.discriminator(x)
        pred = F.softmax(x, dim=1)
        dis_loss = F.cross_entropy(self.label, pred)
        return dis_loss

    def forward(self, x):
        x = self.discriminator(x)
        pred = F.softmax(x, dim=1)
        dis_loss = F.cross_entropy(self.label, pred)
        return dis_loss
