import gzip
import pickle
import numpy as np
import torch
from torch import nn
import random
from tqdm import tqdm
import os
import subprocess
import collections
import igraph
import argparse
import pdb
import pygraphviz as pgv
import sys
from PIL import Image

# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()


'''dataset generation'''


def clamp(minn, maxn, n):
    return max(min(maxn, n), minn)


'''Data preprocessing'''


def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def load_circuit_graphs(topo_num, rand_seed=0):
    # load DAG format CIRCUITs to igraphs
    g_list = []
    random_topo = np.load("./topo_all_1600.npy")
    for i in range(topo_num):
        g = decode_circuit_to_igraph(random_topo[i])
        g_list.append(g)
    graph_args.nvt = 12  # node types
    ng = len(g_list)
    print('# node types: %d' % graph_args.nvt)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng * 0.9)], g_list[int(ng * 0.9):], graph_args


def decode_circuit_to_igraph(random_topo):
    g = igraph.Graph(directed=True)
    g.add_vertices(13)
    g.vs[0]['type'] = 9  # virtual start
    g.vs[12]['type'] = 10  # virtual end
    g.vs[11]['type'] = 11 # gnd

    if random_topo[0]:
        g.vs[1]['type'] = 7  # gm_diff_neg
        g.vs[2]['type'] = 8  # gm_middle_pos
        g.vs[8]['type'] = 7  # gm_middle_neg
    else:
        g.vs[1]['type'] = 8  # gm_diff_pos
        g.vs[2]['type'] = 7  # gm_middle_neg
        g.vs[8]['type'] = 8  # gm_middle_pos
    if random_topo[1]:
        g.vs[3]['type'] = 5
    else:
        g.vs[3]['type'] = 0
    if random_topo[2]:
        if random_topo[0]:
            g.vs[4]['type'] = 6  # gm_diff_neg
        else:
            g.vs[4]['type'] = 5
    else:
        g.vs[4]['type'] = 0
    if random_topo[3]:
        g.vs[5]['type'] = 5
    else:
        g.vs[5]['type'] = 0
    g.vs[6]['type'] = random_topo[4]
    g.vs[7]['type'] = random_topo[5]
    g.vs[9]['type'] = random_topo[6]
    g.vs[10]['type'] = random_topo[7]
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 8)
    g.add_edge(8, 12)
    g.add_edge(0, 3)
    g.add_edge(3, 8)
    g.add_edge(0, 4)
    g.add_edge(4, 12)
    g.add_edge(1, 5)
    g.add_edge(5, 12)
    g.add_edge(1, 6)
    g.add_edge(6, 12)
    g.add_edge(1, 9)
    g.add_edge(9, 11)
    g.add_edge(1, 10)
    g.add_edge(10, 11)
    g.add_edge(11, 12)
    if random_topo[0]:
        g.add_edge(2, 7)
        g.add_edge(7, 12)
    else:
        g.add_edge(1, 7)
        g.add_edge(7, 8)
    return g


def is_same_DAG(g0, g1):
    sign = 1
    label = []
    label_recon = []
    for vo in range(1, g0.vcount() - 1):
        g0_node_type = g0.vs['type'][vo]
        g1_node_type = g1.vs['type'][vo]
        label.append(g0_node_type)
        label_recon.append(g1_node_type)
        if g0_node_type != g1_node_type:
            sign = 0
            break
    return sign
