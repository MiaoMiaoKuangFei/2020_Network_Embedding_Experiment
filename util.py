import optparse
import logging
import re
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import time


def model_choice_dataset_args(usage):
    parser = optparse.OptionParser(usage)
    parser.add_option('-m', dest='m', help='Model', type='str', default='model')
    parser.add_option('-d', dest='d', help='Whole Dataset', type='str', default='foursq2014_TKY_node_format.txt')
    options, args = parser.parse_args()

    return options, args


def log_def(log_file_name="log.log"):
    logging.basicConfig(filename=log_file_name, filemode="w", format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                        level=logging.INFO)


def parse_model_name(model_name):
    if re.match(model_name, '.emb$'):
        return 0  # .emb
    else:
        return 1  # model


def import_model(model_name):
    c = parse_model_name(model_name)
    vertex_id2vec = {}
    if c == 0:
        word_vectors = np.loadtxt(model_name, delimiter=' ')
        for line in word_vectors:
            vertex_id2vec.update({str(int(line[0])): line[1:-1]})
    else:
        model = Word2Vec.load(model_name)
        word_vectors = KeyedVectors.load(model_name)

        for key in word_vectors.wv.vocab.keys():
            vertex_id2vec.update({key: model.wv.__getitem__(key)})
    return vertex_id2vec  # str->list of int


def import_node_label_dict(dsname):
    node_data = np.loadtxt(dsname, delimiter="\t").tolist()
    node_data_dict = {}
    for edge in node_data:
        node_data_dict.update({str(int(edge[0])): str(int(edge[1]))})
        node_data_dict.update({str(int(edge[2])): str(int(edge[3]))})
    del node_data_dict


def import_net(net_path):
    """
    import dataset from net_path
    :return:
    """
    logging.info("start input dataset")
    edge_dict = {}
    time_dict = {}
    io_cost = 0
    try:
        import_net_start = time.time()
        all_edge_list = np.loadtxt(net_path, delimiter='\t')
        all_edge_list = all_edge_list.astype(np.int64)
        for edge in all_edge_list:
            if edge[0] == edge[1]:
                continue
            back_node_accord_edge = edge_dict.get(edge[1])
            front_node_accord_edge = edge_dict.get(edge[0])

            if back_node_accord_edge is None:
                back_node_accord_edge = [[edge[0], edge[2]]]
            else:
                back_node_accord_edge.append([edge[0], edge[2]])

            edge_dict.update({edge[1]: back_node_accord_edge})

            if front_node_accord_edge is None:
                front_node_accord_edge = [[edge[1], edge[2]]]
            else:
                front_node_accord_edge.append([edge[1], edge[2]])

            edge_dict.update({edge[0]: front_node_accord_edge})

            time_accord_edge = time_dict.get(edge[2])

            if time_accord_edge is None:
                time_accord_edge = [edge[0:2].tolist(),
                                    [edge[1], edge[0]]]
            else:
                time_accord_edge.append(edge[0:2].tolist())
                time_accord_edge.append([edge[1], edge[0]])
            time_dict.update({edge[2]: time_accord_edge})

        import_net_end = time.time()
        io_cost = io_cost + (import_net_end - import_net_start)
        logging.info("finish input dataset")
        return edge_dict, time_dict, io_cost
    except Exception as e:
        logging.error("Load dataset error!")
        print(e)
        return edge_dict, time_dict, io_cost
