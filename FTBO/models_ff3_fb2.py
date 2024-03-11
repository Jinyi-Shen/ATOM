import math
import random
import torch
from torch import nn
from torch.nn import functional as F, MSELoss as MSE
import torch.nn.init as init
import numpy as np
import igraph
import pdb

# This file implements several VAE models for DAGs, including SVAE, GraphRNN, DVAE, GCN etc.

'''
    DAG Variational Autoencoder (D-VAE).
'''


class DVAE(nn.Module):
    def __init__(self, max_n, nvt, START_TYPE, END_TYPE, GND_TYPE, POS_TYPE, NEG_TYPE, hs=501, nz=56, vid=True):
        super(DVAE, self).__init__()
        self.max_n = max_n  # maximum number of vertices
        self.nvt = nvt  # number of vertex types
        self.START_TYPE = START_TYPE
        self.END_TYPE = END_TYPE
        self.GND_TYPE = GND_TYPE
        self.POS_TYPE = POS_TYPE
        self.NEG_TYPE = NEG_TYPE
        self.hs = hs  # hidden state size of each vertex
        self.nz = nz  # size of latent representation z
        self.gs = hs  # size of graph state
        self.vid = vid
        self.device = None
        self.n_type = 7

        if self.vid:
            self.vs = hs + max_n  # vertex state size = hidden state + vid
        else:
            self.vs = hs

        # 0. encoding-related
        self.grue_forward_type = nn.GRUCell(nvt, hs)  # encoder GRU
        self.fc1 = nn.Linear(self.gs, nz)  # latent mean
        self.fc2 = nn.Linear(self.gs, nz)  # latent logvar

        # 1. decoding-related
        self.grud_type = nn.GRUCell(nvt, hs)  # decoder GRU
        self.fc3 = nn.Linear(nz, hs)  # from latent z to initial hidden state h0
        self.amp_type = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )
        self.add_ff1 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )
        self.add_ff2 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )
        self.add_ff3 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )
        self.add_fb1 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, self.n_type)
        )
        self.add_fb2 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, self.n_type)
        )
        self.add_ld1 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )
        self.add_ld2 = nn.Sequential(
            nn.Linear(hs, hs * 4),
            nn.ReLU(),
            nn.Linear(hs * 4, hs * 2),
            nn.ReLU(),
            nn.Linear(hs * 2, 2)
        )

        # 2. gate-related
        self.gate_forward = nn.Sequential(
            nn.Linear(self.vs, hs),
            nn.Sigmoid()
        )
        self.mapper_forward = nn.Sequential(
            nn.Linear(self.vs, hs, bias=False),
        )  # disable bias to ensure padded zeros also mapped to zeros

        # 4. other
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.logsoftmax1 = nn.LogSoftmax(1)

    def get_device(self):
        self.device = torch.device('cpu')
        return self.device

    def _get_zeros(self, n, length):
        return torch.zeros(n, length).to(self.get_device())  # get a zero hidden state

    def _get_zero_hidden(self, n=1):
        return self._get_zeros(n, self.hs)  # get a zero hidden state

    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1).to(self.get_device())
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1).to(self.get_device())
        return x

    def _gated(self, h, gate, mapper):
        return gate(h) * mapper(h)

    def _collate_fn(self, G):
        return [g.copy() for g in G]

    def _propagate_to(self, G, v, propagator_type, H=None):
        # propagate messages to vertex index v for all graphs in G
        G = [g for g in G if g.vcount() > v]
        if len(G) == 0:
            return
        if H is not None:
            idx = [i for i, g in enumerate(G) if g.vcount() > v]
            H = H[idx]
        v_types = [g.vs[v]['type'] for g in G]
        X_type = self._one_hot(v_types, self.nvt)
        H_name = 'H_forward'  # name of the hidden states attribute
        H_pred = [[g.vs[x][H_name] for x in g.predecessors(v)] for g in G]
        if self.vid:
            vids = [self._one_hot(g.predecessors(v), self.max_n) for g in G]
            H_pred = [[torch.cat([x[i], y[i:i + 1]], 1) for i in range(len(x))] for x, y in zip(H_pred, vids)]
        gate, mapper = self.gate_forward, self.mapper_forward
        # if h is not provided, use gated sum of v's predecessors' states as the input hidden state
        if H is None:
            max_n_pred = max([len(x) for x in H_pred])  # maximum number of predecessors
            if max_n_pred == 0:
                H = self._get_zero_hidden(len(G))
            else:
                H_pred = [torch.cat(h_pred +
                                    [self._get_zeros(max_n_pred - len(h_pred), self.vs)], 0).unsqueeze(0)
                          for h_pred in H_pred]  # pad all to same length
                H_pred = torch.cat(H_pred, 0)  # batch * max_n_pred * vs
                H = self._gated(H_pred, gate, mapper).sum(1)  # batch * hs
        Hv = propagator_type(X_type, H)

        for i, g in enumerate(G):
            g.vs[v][H_name] = Hv[i:i + 1]
        return Hv

    def _propagate_from(self, G, v, propagator_type, H0=None):
        # perform a series of propagation_to steps starting from v following a topo order
        # assume the original vertex indices are in a topological order
        prop_order = range(v, self.max_n)
        Hv = self._propagate_to(G, v, propagator_type, H0)  # the initial vertex
        for v_ in prop_order[1:]:
            self._propagate_to(G, v_, propagator_type)
        return Hv

    def _update_v(self, G, v, H0=None):
        # perform a forward propagation step at v when decoding to update v's state
        self._propagate_to(G, v, self.grud_type, H0)
        return

    def _get_vertex_state(self, G, v):
        # get the vertex states at v
        Hv = []
        for g in G:
            if v >= g.vcount():
                hv = self._get_zero_hidden()
            else:
                hv = g.vs[v]['H_forward']
            Hv.append(hv)
        Hv = torch.cat(Hv, 0)
        return Hv

    def _get_graph_state(self, G):
        # get the graph states
        # when decoding, use the last generated vertex's state as the graph state
        # when encoding, use the ending vertex state or unify the starting and ending vertex states
        Hg = []
        for g in G:
            hg = g.vs[g.vcount() - 1]['H_forward']
            Hg.append(hg)
        Hg = torch.cat(Hg, 0)
        return Hg

    def encode(self, G):
        # encode graphs G into latent vectors
        if type(G) != list:
            G = [G]
        self._propagate_from(G, 0, self.grue_forward_type, H0=self._get_zero_hidden(len(G)))
        Hg = self._get_graph_state(G)
        mu, logvar = self.fc1(Hg), self.fc2(Hg)
        return mu, logvar

    def reparameterize(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z, stochastic=True):
        # decode latent vectors z back to graphs
        # if stochastic=True, stochastically sample each action from the predicted distribution;
        # otherwise, select argmax action deterministically.
        H0 = self.tanh(self.fc3(z))  # or relu activation
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex()
            g.vs[0]['type'] = self.START_TYPE
        self._update_v(G, 0, H0)
        Hv = self._get_graph_state(G)
        type = self.amp_type(Hv)
        amp_types = torch.argmax(type, 1)
        amp_types = amp_types.flatten().tolist()
        for idx in range(1, 13):
            Hv = self._get_graph_state(G)
            if idx == 11 or idx == 12 or idx == 1 or idx == 2 or idx == 8:
                pass
            elif idx == 3:
                type_scores = self.add_ff1(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(2), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            elif idx == 4:
                type_scores = self.add_ff2(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(2), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            elif idx == 5:
                type_scores = self.add_ff3(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(2), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            elif idx == 6:
                type_scores = self.add_fb1(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.n_type), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            elif idx == 7:
                type_scores = self.add_fb2(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(self.n_type), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            elif idx == 9:
                type_scores = self.add_ld1(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(2), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            else:
                type_scores = self.add_ld2(Hv)
                if stochastic:
                    type_probs = F.softmax(type_scores, 1).cpu().detach().numpy()
                    new_types = [np.random.choice(range(2), p=type_probs[i])
                                 for i in range(len(G))]
                else:
                    new_types = torch.argmax(type_scores, 1)
                    new_types = new_types.flatten().tolist()
            for i, g in enumerate(G):
                g.add_vertex()
                if idx == 1:
                    g.add_edge(0, 1)
                    if amp_types[i]:
                        g.vs[idx]['type'] = self.NEG_TYPE
                    else:
                        g.vs[idx]['type'] = self.POS_TYPE
                elif idx == 2:
                    g.add_edge(1, 2)
                    if amp_types[i]:
                        g.vs[idx]['type'] = self.POS_TYPE
                    else:
                        g.vs[idx]['type'] = self.NEG_TYPE
                elif idx == 3:
                    g.add_edge(0, 3)
                    if new_types[i]:
                        g.vs[idx]['type'] = 5
                    else:
                        g.vs[idx]['type'] = 0
                elif idx == 4:
                    g.add_edge(0, 4)
                    if new_types[i]:
                        if amp_types[i]:
                            g.vs[idx]['type'] = 6
                        else:
                            g.vs[idx]['type'] = 5
                    else:
                        g.vs[idx]['type'] = 0
                elif idx == 5:
                    g.add_edge(1, 5)
                    if new_types[i]:
                        g.vs[idx]['type'] = 5
                    else:
                        g.vs[idx]['type'] = 0
                elif idx == 6:
                    g.add_edge(1, 6)
                    g.vs[idx]['type'] = new_types[i]
                elif idx == 7:
                    g.vs[idx]['type'] = new_types[i]
                    if amp_types[i]:
                        g.add_edge(2, 7)
                    else:
                        g.add_edge(1, 7)
                elif idx == 8:
                    g.add_edge(2, 8)
                    g.add_edge(3, 8)
                    if amp_types[i]:
                        g.vs[idx]['type'] = self.NEG_TYPE
                    else:
                        g.vs[idx]['type'] = self.POS_TYPE
                        g.add_edge(7, 8)
                elif idx == 9:
                    g.add_edge(1, 9)
                    g.vs[idx]['type'] = new_types[i]
                elif idx == 10:
                    g.add_edge(2, 10)
                    g.vs[idx]['type'] = new_types[i]
                elif idx == 11:
                    g.add_edge(9, 11)
                    g.add_edge(10, 11)
                    g.vs[idx]['type'] = self.GND_TYPE
                elif idx == 12:
                    g.add_edge(4, 12)
                    g.add_edge(5, 12)
                    g.add_edge(6, 12)
                    g.add_edge(8, 12)
                    g.add_edge(11, 12)
                    g.vs[idx]['type'] = self.END_TYPE
                    if amp_types[i]:
                        g.add_edge(7, 12)
            self._update_v(G, idx)

        for g in G:
            del g.vs['H_forward']  # delete hidden states to save GPU memory
        return G

    def loss(self, mu, logvar, G_true, beta=0.005):
        # compute the loss of decoding mu and logvar to true graphs using teacher forcing
        # ensure when computing the loss of step i, steps 0 to i-1 are correct
        z = self.reparameterize(mu, logvar)
        H0 = self.tanh(self.fc3(z))  # or relu activation
        G = [igraph.Graph(directed=True) for _ in range(len(z))]
        for g in G:
            g.add_vertex()
            g.vs[0]['type'] = self.START_TYPE
        self._update_v(G, 0, H0)
        softmax_error = torch.Tensor([0.0]).float().to(self.get_device())
        true_amp_types = []
        for g in G_true:
            if g.vs[1]['type'] == 7:
                true_amp_types.append(1)
            else:
                true_amp_types.append(0)
        Hv = self._get_graph_state(G)
        type_scores = self.amp_type(Hv)
        softmax_error += self.logsoftmax1(type_scores)[np.arange(len(G)), true_amp_types].sum().to(
            self.get_device())
        for v_true in range(1, 13):
            if 3 <= v_true <= 7 or v_true == 9 or v_true == 10:
                Hv = self._get_graph_state(G)
                if v_true == 3:
                    type_scores0 = self.add_ff1(Hv)
                    type_scores = torch.zeros((len(G), self.n_type))
                    type_scores[:, 0] = type_scores0[:, 0]
                    type_scores[:, 5] = type_scores0[:, 1]
                elif v_true == 4:
                    type_scores0 = self.add_ff2(Hv)
                    type_scores = torch.zeros((len(G), self.n_type))
                    type_scores[:, 0] = type_scores0[:, 0]
                    for i in range(len(true_amp_types)):
                        if true_amp_types[i]:
                            type_scores[i, 6] = type_scores0[i, 1]
                        else:
                            type_scores[i, 5] = type_scores0[i, 1]
                elif v_true == 5:
                    type_scores0 = self.add_ff3(Hv)
                    type_scores = torch.zeros((len(G), self.n_type))
                    type_scores[:, 0] = type_scores0[:, 0]
                    type_scores[:, 5] = type_scores0[:, 1]
                elif v_true == 6:
                    type_scores = self.add_fb1(Hv)
                elif v_true == 7:
                    type_scores = self.add_fb2(Hv)
                elif v_true == 9:
                    type_scores = self.add_ld1(Hv)
                else:
                    type_scores = self.add_ld2(Hv)
                # vertex log likelihood
                true_types = [g.vs[v_true]['type'] for g in G_true]
                softmax_error += self.logsoftmax1(type_scores)[np.arange(len(G)), true_types].sum().to(
                    self.get_device())

            for i, g in enumerate(G):
                g.add_vertex(type=int(G_true[i].vs[v_true]['type']))
                if v_true == 1:
                    g.add_edge(0, 1)
                elif v_true == 2:
                    g.add_edge(1, 2)
                elif v_true == 3:
                    g.add_edge(0, 3)
                elif v_true == 4:
                    g.add_edge(0, 4)
                elif v_true == 5:
                    g.add_edge(1, 5)
                elif v_true == 6:
                    g.add_edge(1, 6)
                elif v_true == 7:
                    if true_amp_types[i]:
                        g.add_edge(2, 7)
                    else:
                        g.add_edge(1, 7)
                elif v_true == 8:
                    g.add_edge(2, 8)
                    g.add_edge(3, 8)
                    if not true_amp_types[i]:
                        g.add_edge(7, 8)
                elif v_true == 9:
                    g.add_edge(1, 9)
                elif v_true == 10:
                    g.add_edge(2, 10)
                elif v_true == 11:
                    g.add_edge(9, 11)
                    g.add_edge(10, 11)
                elif v_true == 12:
                    g.add_edge(4, 12)
                    g.add_edge(5, 12)
                    g.add_edge(6, 12)
                    g.add_edge(8, 12)
                    g.add_edge(11, 12)
                    if true_amp_types[i]:
                        g.add_edge(7, 12)
            self._update_v(G, v_true)

        res = - softmax_error  # convert likelihood to loss
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return res + beta * kld, res, -softmax_error, kld

    def encode_decode(self, G):
        mu, logvar = self.encode(G)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def forward(self, G):
        mu, logvar = self.encode(G)
        loss, recon, softmax_error, kld = self.loss(mu, logvar, G)
        return loss, recon, softmax_error, kld

    def generate_sample(self, n):
        sample = torch.randn(n, self.nz).to(self.get_device())
        G = self.decode(sample)
        return G