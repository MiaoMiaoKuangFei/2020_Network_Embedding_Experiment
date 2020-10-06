import optparse
import os
import random
import util


def create_neg_sample(ratio, train, test, nodes):
    pos_sample = train.copy()
    pos_sample.extend(test)
    hashable_pos_sample = ['_'.join(edge) for edge in pos_sample]
    hashable_pos_sample = set(hashable_pos_sample)
    hashable_test = ['_'.join(edge) for edge in test]
    hashable_test = set(hashable_test)
    neg_samp = []
    candi_sample = []  # 这部分的sample是迫不得已再用
    while True:
        if len(neg_samp) % 1000 == 0:
            print("当前进度为：", len(neg_samp) / (len(test) * ratio))
        # print(len(neg_samp))
        n1, n2 = random.sample(nodes, 2)
        if n1 != n2 and '_'.join([n1, n2]) not in hashable_pos_sample and '_'.join([n2, n1]) not in hashable_pos_sample:
            neg_samp.append([n1, n2])
        else:
            if n1 != n2 and '_'.join([n1, n2]) not in hashable_test and '_'.join([n2, n1]) not in hashable_test:
                candi_sample.append([n1, n2])

        if len(test) * ratio < len(neg_samp):
            break

    if len(test) * ratio > len(neg_samp):
        ext = len(test) * ratio - len(neg_samp)
        if ext > len(candi_sample):
            neg_samp.extend(candi_sample)
            print("负例仍然不足，差距是", len(pos_sample) * ratio - len(neg_samp))
        else:
            neg_samp.extend(candi_sample[:ext])
    else:
        neg_samp = neg_samp[:len(test) * ratio]

    return neg_samp


def save_neg_sample(neg, path):
    if not os.path.exists(path):
        open(path, 'w').close()

    with open(path, 'a') as f:
        for edge in neg:
            f.write('\t'.join(edge) + '\n')


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

# -t experiment/emb/link_prediction_folder/tky/100000.txt -s experiment/emb/link_prediction_folder/tky/200000.txt -o experiment/emb/link_prediction_folder/tky/1_2_neg.txt
# -t experiment/emb/link_prediction_folder/tky/200000.txt -s experiment/emb/link_prediction_folder/tky/300000.txt -o experiment/emb/link_prediction_folder/tky/2_3_neg.txt
# -t experiment/emb/link_prediction_folder/tky/300000.txt -s experiment/emb/link_prediction_folder/tky/400000.txt -o experiment/emb/link_prediction_folder/tky/3_4_neg.txt
# -t experiment/emb/link_prediction_folder/tky/400000.txt -s experiment/emb/link_prediction_folder/tky/500000.txt -o experiment/emb/link_prediction_folder/tky/4_5_neg.txt
# -t experiment/emb/link_prediction_folder/tky/500000.txt -s experiment/emb/link_prediction_folder/tky/555438.txt -o experiment/emb/link_prediction_folder/tky/5_end_neg.txt