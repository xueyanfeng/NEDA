from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import pandas as pd
import csv

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from GraphSage.encoders import Encoder
from GraphSage.aggregators import MeanAggregator
from GraphSage.model import SupervisedGraphSage
from utils_NEDA import load_data, EarlyStopping, accuracy, get_excel_name

parser = argparse.ArgumentParser("")
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--gcn', action='store_true', default=False,
                    help='Determine the aggregation method.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--early-stop', action='store_true', default=True,
                    help="indicates whether to use early stop or not")
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--similarity', action='store_true', default=True, help='For comparison with baseline')
parser.add_argument('--infected_number', type=int, default=15, help="Number of infected individuals")
parser.add_argument('--is_copy', action='store_true', default=False, help='Whether the attributes of the infected person are copied')
parser.add_argument('--sample1', type=int, default=5, help="Number of neighbors for first-order sampling")
parser.add_argument('--sample2', type=int, default=10, help="Number of neighbors for second-order sampling")
parser.add_argument('--dataset', type=str, default='wisconsin')
args = parser.parse_args()

def experiment(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset = args.dataset
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.cuda_device)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)

    # Load data
    feat_data, labels, adj_lists, train, val, test, extended_neighborhood_coefficient \
        = load_data(dataset, args.infected_number, args.similarity, args.is_copy)

    features = nn.Embedding(*feat_data.shape)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=args.cuda, gcn=args.gcn, similarity=args.similarity)
    enc1 = Encoder(features, feat_data.shape[1], args.hidden, adj_lists, agg1, args.sample1, gcn=args.gcn, cuda=args.cuda)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=args.cuda, gcn=args.gcn, similarity=args.similarity)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), args.hidden, args.hidden, adj_lists, agg2, args.sample2,
                   base_model=enc1, gcn=args.gcn, cuda=args.cuda)

    graphsage = SupervisedGraphSage(labels.max().item() + 1, enc2)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        t0 = time.time()
        graphsage.train()
        train_batch_nodes = train  # train[:256]
        random.shuffle(train)

        # forward
        train_logits = graphsage(train_batch_nodes)
        train_loss = loss_fcn(train_logits, Variable(torch.LongTensor(labels[np.array(train_batch_nodes)])))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_acc = accuracy(train_logits, labels[train_batch_nodes])

        graphsage.eval()
        val_batch_nodes = val
        with torch.no_grad():
            val_logits = graphsage(val_batch_nodes)
        val_acc = accuracy(val_logits, labels[val_batch_nodes])
        if args.early_stop:
            if stopper.step(val_acc, graphsage):
                break
        if args.fastmode:
            continue
        else:
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " ValAcc {:.4f}".format(epoch, time.time() - t0, train_loss.item(), train_acc, val_acc))

    print()
    if args.early_stop:
        graphsage.load_state_dict(torch.load('es_checkpoint.pt'))
    test_logits = graphsage(test)
    test_acc = accuracy(test_logits, labels[test])
    print("Test Accuracy {:.4f}".format(test_acc))
    return test_acc, extended_neighborhood_coefficient

def experiment_average(number_experiments):
    t0 = time.time()
    out_f = open('{}{}_{}_{}_{}_{}.csv'.format(result,
                                               args.dataset,
                                               get_excel_name(args.similarity, args.is_copy),
                                               args.sample1,
                                               args.sample2,
                                               args.infected_number), 'w', newline='')
    writer = csv.writer(out_f)
    writer.writerow(["s1", "s2", "s0", "accuracy"])

    acc_result = []
    enc_list = []
    for i in range(number_experiments):
        args.seed = i
        print(args)
        test_acc, extended_neighborhood_coefficient = experiment(args)

        enc_list.append(pd.Series(extended_neighborhood_coefficient).to_frame(str(i)))
        acc_result.append(test_acc)

    print()
    print("{}_{}_Average_accuracy:{} | Time(s) {:.4f}".format(args.dataset,
                                                              get_excel_name(args.similarity, args.is_copy),
                                                              np.array(acc_result).mean(),
                                                              time.time() - t0))
    writer.writerow([args.sample1, args.sample2, args.infected_number, np.array(acc_result).mean()])
    out_f.close()

    for i in range(1, number_experiments):
        enc_list[0] = pd.merge(enc_list[0], enc_list[i], left_index=True, right_index=True, how='outer')
    enc_list[0].to_excel(
        '{}{}/{}_enc.xlsx'.format(plot, args.dataset, get_excel_name(args.similarity, args.is_copy)))
    return np.array(acc_result).mean()


if __name__ == '__main__':
    number_experiments = 10
    result = "./results/"
    plot = "./plot/"

    # args = parser.parse_args(['--no-cuda','--is_copy', '--dataset','wisconsin','--infected_number','37','--sample1','21','--sample2','9'])
    experiment_average(number_experiments)

    print("finished")