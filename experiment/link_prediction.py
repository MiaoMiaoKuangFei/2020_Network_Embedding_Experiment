import optparse
import util
import numpy as np
from sklearn.metrics import roc_auc_score


def get_cosine_distance(node1_vec, node2_vec):
    num = np.float(np.sum(node1_vec * node2_vec))
    denom = np.linalg.norm(node1_vec) * np.linalg.norm(node2_vec)
    return num / denom


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-s', dest='s', help='Test-set', type='str', default='test.txt')
    parser.add_option('-o', dest='o', help='Neg-set', type='str', default='/result')
    parser.add_option('-m', dest='m', help='model', type='str', default='/emb/model')
    options, args = parser.parse_args()

    id2vec = util.import_model(options.m)
    pos_edges, nd1 = util.import_edges(options.s, '5')
    neg_edges, nd2 = util.import_edges(options.o, '3')
    st_vec = []
    sec_vec = []
    label = []
    print("处理正例")
    for edge in pos_edges:
        if edge[0] in id2vec and edge[1] in id2vec:
            st_vec.append(id2vec.get(edge[0]))
            sec_vec.append(id2vec.get(edge[1]))
            label.append(1)

    print("处理负例")
    for edge in neg_edges:
        if edge[0] in id2vec and edge[1] in id2vec:
            st_vec.append(id2vec.get(edge[0]))
            sec_vec.append(id2vec.get(edge[1]))
            label.append(0)

    st_vec = np.array(st_vec)
    sec_vec = np.array(sec_vec)

    cos_arr = []
    for i in range(len(st_vec)):
        cos_arr.append(get_cosine_distance(st_vec[i], sec_vec[i]))
    print("AUC", roc_auc_score(label, cos_arr))
# -o experiment/emb/link_prediction_folder/tky/1_2_neg.txt -s experiment/emb/link_prediction_folder/tky/200000.txt -m experiment/emb/link_prediction_folder/tky/9_model
# -o experiment/emb/link_prediction_folder/tky/2_3_neg.txt -s experiment/emb/link_prediction_folder/tky/300000.txt -m experiment/emb/link_prediction_folder/tky/19_model
# -o experiment/emb/link_prediction_folder/tky/3_4_neg.txt -s experiment/emb/link_prediction_folder/tky/400000.txt -m experiment/emb/link_prediction_folder/tky/29_model
# -o experiment/emb/link_prediction_folder/tky/4_5_neg.txt -s experiment/emb/link_prediction_folder/tky/500000.txt -m experiment/emb/link_prediction_folder/tky/39_model
# -o experiment/emb/link_prediction_folder/tky/5_end_neg.txt -s experiment/emb/link_prediction_folder/tky/555438.txt -m experiment/emb/link_prediction_folder/tky/49_model