#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : online_CTDNEs.py
# @author : Zhutian Lin
# @date   : 2020/9/17
# @version: 2.0
# @desc   : Online-CTDNEs Alg
import time
import numpy as np
from gensim.models import Word2Vec
import logging
import optparse
import os
import util


def sample_next_edge(curr_edge, legal_neighbour_edge):
    """
    sample next edge by linear probability
    :param curr_edge: Edge start the walk from
    :param legal_neighbour_edge: adjacent edges from now one
    :return: a chosen legal next edge
    """
    mid_node = curr_edge[1]
    if len(legal_neighbour_edge) == 0:
        return []

    prob_array = np.array(list(range(len(legal_neighbour_edge)))) + 1
    prob_array = prob_array / np.sum(prob_array)
    index = np.random.choice(range(0, len(legal_neighbour_edge)), p=prob_array)
    sample = [mid_node, legal_neighbour_edge[index][0], legal_neighbour_edge[index][1]]

    return sample


class online_ctdns:
    """
    Online CTDNEs-Alg
    """

    def __init__(self, r, l, w, d, granularity, result_path, net_path):
        self.edge_dict = {}  # vertex -> edge
        self.net = []  # The whole net
        self.result_path = result_path
        self.net_path = net_path
        self.all_edge_list = []
        self.time_new_edge_dict = {}  # time -> edge

        self.N = len(self.edge_dict.keys())
        self.r = r  # Count of walks from curr edge
        self.l = l  # Upper bound of a walk
        self.w = w  # Lower bound of a walk
        self.d = d  # Dim of emb vec
        self.granularity = granularity  # Granularity of each window(For batch update)
        self.io_cost = 0.0

    def import_net(self):
        """
        import dataset from self.net_path
        :return:
        """
        self.edge_dict, self.time_new_edge_dict, io_cost = util.import_net(self.net_path)
        self.io_cost = self.io_cost + io_cost

    def batch_update(self):
        """
        Batch update along the time stream according to the paper
        :return:
        """
        time_list = [each_time for each_time in self.time_new_edge_dict]
        for i in range(len(time_list) // self.granularity):
            logging.info("Current Progress :" + str(i / (len(time_list) // self.granularity)))
            if (i + 1) * self.granularity >= len(time_list):
                time_slice = time_list[i * self.granularity:len(time_list)]
            else:
                time_slice = time_list[i * self.granularity:(i + 1) * self.granularity]
            logging.info("The time scope is from " + str(time_slice[0]) + " to " + str(time_slice[-1]))
            for j, each_time in enumerate(time_slice):
                self.random_walk(each_time, walk_num=self.r)
            try:
                texts = self.load_walk_set()
                model = Word2Vec(texts, sg=1, size=self.d, window=10, min_count=0, workers=8)
                model.save(self.result_path + '/model/' + str(i) + '_model')
            except Exception as e:
                logging.info("walk is too less to train")
                print(e)

    def random_walk(self, t, walk_num=30):
        """
        Walk and save to txt
        :param t: timestamp
        :param walk_num: count of walks
        """
        edge_list_at_new_time = np.array(self.time_new_edge_dict.get(t))
        curr_new_edge_arr = np.insert(edge_list_at_new_time, 2, values=t, axis=1)

        walks = []
        for each_new_edge in curr_new_edge_arr:
            for i in range(walk_num):
                walk_index = self.reverse_temporal_walk(each_new_edge.tolist())
                walk_index.reverse()
                if self.w < len(walk_index) < self.l and walk_index != []:
                    walks.append(str(walk_index).replace('[', '').replace(']', ''))
        if walks:
            batch_walks_str = '\n'.join(walks)
            io_start = time.time()
            with open(self.result_path + "/walk/walk.txt", 'a') as f:
                f.write(batch_walks_str)
                f.write('\n')
            io_end = time.time()
            self.io_cost = self.io_cost + io_end - io_start

    def load_walk_set(self):
        """
        Load walks and transfer into a list
        :return: 2-dim walks list
        """
        load_walk_start = time.time()
        dataset = []
        with open(self.result_path + "/walk/walk.txt", 'r') as f:
            sourceInLine = f.readlines()
            for line in sourceInLine:
                dataset.append(line.strip('\n').split(', '))
        load_walk_end = time.time()
        self.io_cost = self.io_cost + (float(load_walk_end) - float(load_walk_start))
        return dataset

    def reverse_temporal_walk(self, start_edge):
        """
        Walk a path reversely
        :return: Reverse temporal walk
        """
        curr_walk_index = start_edge[0:2]  # Fetch start_edge[0 and 1]
        curr_edge = start_edge
        for p in range(1, self.l - 1):

            legal_neighbour_edge = self.find_all_legal_adjacent_edges(curr_edge)
            if legal_neighbour_edge != [] and len(legal_neighbour_edge) > 0:
                legal_neighbour_edge.sort(key=lambda x: x[1])
                choose_neighbour = sample_next_edge(curr_edge, legal_neighbour_edge)
                curr_walk_index.append(choose_neighbour[1])
                curr_edge = choose_neighbour
            else:
                return curr_walk_index

        return curr_walk_index

    def find_all_legal_adjacent_edges(self, curr_edge):
        """
        Find all edges adjacent with current one,which satisfy the constraint of time
        :return A list of adjacent edges
        """
        next_node = curr_edge[1]
        neighbour = self.edge_dict.get(next_node)
        curr_time = curr_edge[2]
        if neighbour is None:
            return []
        else:
            legal_neighbour = [shift for shift in neighbour if shift[1] < curr_time]
            return legal_neighbour


if __name__ == '__main__':
    usage = "Online CTDNEs params"
    parser = optparse.OptionParser(usage)
    parser.add_option('-r', dest='r', help='Num of walks from each vertex', type='int', default=10)
    parser.add_option('-w', dest='w', help='Lower Bound of the path length', type='int', default=5)
    parser.add_option('-l', dest='l', help='Upper Bound of the path length', type='int', default=80)
    parser.add_option('-d', dest='d', help='Dimension of each embedding vector', type='int', default=128)
    parser.add_option('-g', dest='g', help='Granularity for batch update', type='int', default=10000)
    parser.add_option('-s', dest='s', help='Path of Dataset', type='string', default='foursq2014_TKY_node_format.txt')
    parser.add_option('-j', dest='j', help='Dir of result', type='string', default='result')
    options, args = parser.parse_args()

    #  initialize path and dir and file
    if not os.path.isdir(options.j):
        os.mkdir(options.j)

    if not os.path.isdir(options.j + "/walk"):
        os.mkdir(options.j + "/walk")

    if not os.path.isdir(options.j + "/model"):
        os.mkdir(options.j + "/model")

    if not os.path.exists(options.j + "/walk/walk.txt"):
        open(options.j + "/walk/walk.txt", 'w').close()

    util.log_def(options.j + "/log.log")

    logging.info("input：" + str(options))
    start = float(time.time())
    oc = online_ctdns(r=options.r,
                      w=options.w,
                      l=options.l,
                      d=options.d,
                      granularity=options.g,
                      result_path=options.j,
                      net_path=options.s)

    oc.import_net()
    oc.batch_update()
    end = float(time.time())

    logging.info("Total time(include I/O time loss)" + str(end - start))
    logging.info("I/O time loss" + str(oc.io_cost))
    logging.info("Total time(exclude I/O time loss)" + str(end - start - oc.io_cost))
