import networkx as nx
import numpy as np
import torch
import os
import math
import adj_set_nettorkx

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def get_excel_name(similarity, is_copy):
    if similarity and is_copy:
        return "NEDA_star"
    else:
        return "NEDA"

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')

def load_data(data_name,infected_number,sim ,is_copy,train_percentage=0.6, val_percentage=0.2):
    graph_adjacency_list_file_path = os.path.join('data', data_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('data', data_name,
                                                            f'out1_node_feature_label.txt')
    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if data_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))
            G.add_edge(int(line[1]), int(line[0]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    assert (features == 1).sum() == len(features.nonzero()[0])
    features = adj_set_nettorkx.preprocess_features(features)
    assert len(list(G.edges())) == adj.nnz

    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.to_directed()


    adj_lists = adj_set_nettorkx.get_adj_set_nettorkx(G, infected_number, sim, is_copy)

    extended_neighborhood_coefficient = adj_set_nettorkx.get_extended_neighborhood_coefficient(G,adj_lists)

    assert (train_percentage < 1.0 and val_percentage < 1.0 and train_percentage + val_percentage < 1.0)
    rand_indices = np.random.permutation(labels.shape[0])
    train = list(rand_indices[:math.ceil(len(rand_indices) * train_percentage)])
    val = list(rand_indices[math.ceil(len(rand_indices) * train_percentage):math.ceil(
        len(rand_indices) * (train_percentage + val_percentage))])
    test = list(rand_indices[math.ceil(len(rand_indices) * (train_percentage + val_percentage)):])
    return torch.tensor(features, dtype=torch.float32), \
           torch.tensor(labels, dtype=torch.int64), adj_lists, train, val, test, \
           extended_neighborhood_coefficient

if __name__ == '__main__':
    dataset = "texas"
    seed = 0
    # Load data
    feat_data, labels, adj_lists, train, val, test, extended_neighborhood_coefficient = load_data(dataset, 15, True, True)
    print(adj_lists)
