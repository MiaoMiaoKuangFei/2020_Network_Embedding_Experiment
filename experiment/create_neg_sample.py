import optparse
import os

import numpy as np
import util


def create_neg_sample(ratio, train, test, nodes):
    pos_sample = train.union(test)
    neg_samp = []
    candi_sample = []  # 这部分的sample是迫不得已再用
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2 and [n1, n2] not in pos_sample and [n2, n1] not in pos_sample:
                neg_samp.append([n1, n2])
            else:
                if n1 != n2 and [n1, n2] not in test and [n2, n1] not in test:
                    candi_sample.append([n1, n2])
    if len(pos_sample) * ratio > len(neg_samp):
        ext = len(pos_sample) * ratio - len(neg_samp)
        if ext > len(candi_sample):
            neg_samp.extend(candi_sample)
            print("负例仍然不足，差距是", len(pos_sample) * ratio - len(neg_samp))
        else:
            neg_samp.extend(candi_sample[:ext])

    return neg_samp


def save_neg_sample(neg, path):
    if not os.path.exists(path):
        open(path, 'w').close()

    with open(path, 'a') as f:
        for edge in neg:
            f.write('\t'.join(edge))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-t', dest='t', help='Train-set', type='str', default='train.txt')
    parser.add_option('-s', dest='s', help='Test-set', type='str', default='test.txt')
    parser.add_option('-o', dest='o', help='Output', type='str', default='/result')
    parser.add_option('-r', dest='r', help='ratio', type='int', default=1)
    options, args = parser.parse_args()

    train_true, train_nodes = util.import_edges(options.t)
    test_true, test_nodes = util.import_edges(options.s)

    neg_sample = create_neg_sample(options.r, train_true, test_true, train_nodes.union(test_nodes))
    save_neg_sample(neg_sample, options.o)
