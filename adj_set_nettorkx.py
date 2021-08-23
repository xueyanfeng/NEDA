from __future__ import division
from __future__ import print_function

from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cosine as cos
import numpy as np
import scipy.sparse as sp

def is_IS(IS_link_dict,status):
    IS_matrix = np.array(list(IS_link_dict.values()))
    for i in range(IS_matrix.shape[0]):
        if status[IS_matrix[i][0]] != 0 or status[IS_matrix[i][1]] != 1:
            IS_link_dict.pop('{}-{}'.format(IS_matrix[i][0],IS_matrix[i][1]))
    return True

def SI_sampling(G,seed_node,sample_number,is_copy):
    sampled_nodes = set()
    ne_G = G.copy()
    status = defaultdict(lambda: 1)  #: "1" stands for the susceptible
    status[seed_node] = 0  #: "0" stands for the infected
    neib = list(nx.neighbors(ne_G, seed_node))

    IS_link_dict = dict(zip(["{}-{}".format(seed_node,n) for n in neib],[(seed_node,n) for n in neib]))
    # Ensure that the source nodes are the infected and the target nodes are the susceptible

    for j in range(sample_number):
        assert is_IS(IS_link_dict,status)
        prob = [1-cos(G.nodes[s]['features'],G.nodes[t]['features']) for s,t in list(IS_link_dict.values())]
        if np.array(prob).sum():
            p = np.array(prob)
            p /= p.sum()
        else:
            p = np.ones(len(prob), dtype=np.float32)
            p /= p.sum()
        if list(IS_link_dict.keys()):
            index = np.random.choice(list(IS_link_dict.keys()),1,p=p)[0]
        else:
            return sampled_nodes
        infected_node = eval(index.split("-")[-1])
        if is_copy:
            G.nodes[infected_node]['features'] = G.nodes[seed_node]['features']   # the attributes of the infected person are copied
        sampled_nodes.add(infected_node)
        status[infected_node] = 0
        IS_link_dict.pop(index)
        neib = list(nx.neighbors(ne_G, infected_node))
        neib = [n for n in neib if status[n]]
        IS_link_dict.update(dict(zip(["{}-{}".format(infected_node,n) for n in neib],[(infected_node,n) for n in neib])))

    assert isinstance(sampled_nodes,set)
    return sampled_nodes

def get_adj_set_nettorkx(G,infected_number,sim,is_copy):
    adj_lists = defaultdict(set)
    if sim:
        for i in list(G.nodes()):
            adj_lists[i] = SI_sampling(G, i,infected_number,is_copy)
    else:
        for i in list(G.edges()):
            adj_lists[i[0]].add(i[1])
    return adj_lists

def get_extended_neighborhood_coefficient(G,adj_lists):
    enc_dict = {}
    for node,neigh in adj_lists.items():
        enc_dict[node] = np.array([nx.shortest_path_length(G=G,source=node,target=n) for n in list(neigh)],dtype = np.float32).sum()/len(neigh)
    return enc_dict

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = (rowsum==0)*1+rowsum
    r_inv = np.power(rowsum.astype(np.float32), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features