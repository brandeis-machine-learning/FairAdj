# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fair Adjacency Graph Embedding")

    # experiment
    parser.add_argument('--seed', type=int, default=1, help="seed")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--test_ratio', type=float, default=0.1, help='proportion of testing edges')

    # model
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

    # optimize
    parser.add_argument('--outer_epochs', type=int, default=4, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--eta', type=float, default=0.2, help='Learning rate for adjacency matrix.')
    parser.add_argument('--T1', type=int, default=50)
    parser.add_argument('--T2', type=int, default=20)
    parser.add_argument('--eq', type=bool, default=False, help='Set to True for Oklahoma97 and UNC28')

    return parser.parse_args()
