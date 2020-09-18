#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @file   : online_CTDNEs.py
# @author : Zhutian Lin
# @date   : 2020/9/17
# @version: 2.0
# @desc   : 复现Online-CTDNEs算法
import time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import logging
import optparse
from concurrent.futures import ThreadPoolExecutor
import os


def sample_next_edge(curr_edge, legal_neighbour_edge):
    """
    按时间线性选择下一条边
    :param curr_edge: 当前的边
    :param legal_neighbour_edge: 邻边list
    :return: 按概率取出一个符合要求的边,如果没有就返回为[]这个需要注意了！！
    """
    mid_node = curr_edge[1]  # 这个curr_edge的node2充当中继点的作用
    if len(legal_neighbour_edge) == 0:
        return []

    prob_array = np.array(list(range(len(legal_neighbour_edge)))) + 1
    prob_array = prob_array / np.sum(prob_array)
    index = np.random.choice(range(0, len(legal_neighbour_edge)), p=prob_array)  # 这里先写死为线性的，之后其他的再做实验
    sample = [mid_node, legal_neighbour_edge[index][0], legal_neighbour_edge[index][1]]  # 合并为符合条件的新边

    return sample


class online_ctdns:
    def __init__(self, r, l, w, d, granularity, result_path,
                 net_path="foursq2014_TKY_node_format.txt"):
        self.edge_dict = {}  # 以后节点为索引的传播路径
        self.net = []  # 整个原来的图
        self.result_path = result_path
        self.net_path = net_path
        self.all_edge_list = []
        self.time_new_edge_dict = {}  # 以时间为索引，边为value的dict

        self.N = len(self.edge_dict.keys())
        self.r = r  # 以每个节点为起始位置应该走多长，本版本中没有用到
        self.l = l  # 一个行走最长是多少
        self.w = w  # 一个行走最短是多少
        self.d = d  # embedding向量维度
        self.granularity = granularity  # 切分粒度
        self.io_cost = 0.0

    def import_net(self):
        """
        import dataset from self.net_path
        :return:
        """
        logging.info("start input dataset")
        try:
            import_net_start = time.time()
            self.all_edge_list = np.loadtxt(self.net_path, delimiter='\t')
            self.all_edge_list = self.all_edge_list.astype(np.int64)
            for edge in self.all_edge_list:
                if edge[0] == edge[1]:
                    continue
                #  输入的时候默认是有向边，现在修改成无向边
                back_node_accord_edge = self.edge_dict.get(edge[1])  # 入度点对应的边
                front_node_accord_edge = self.edge_dict.get(edge[0])  # 出度点对应的边

                #  对点操作
                if back_node_accord_edge is None:
                    back_node_accord_edge = [[edge[0], edge[2]]]
                else:
                    back_node_accord_edge.append([edge[0], edge[2]])

                self.edge_dict.update({edge[1]: back_node_accord_edge})

                if front_node_accord_edge is None:
                    front_node_accord_edge = [[edge[1], edge[2]]]
                else:
                    front_node_accord_edge.append([edge[1], edge[2]])

                self.edge_dict.update({edge[0]: front_node_accord_edge})

                # 对时间操作
                time_accord_edge = self.time_new_edge_dict.get(edge[2])

                if time_accord_edge is None:
                    time_accord_edge = [edge[0:2].tolist(),
                                        [edge[1], edge[0]]]
                else:
                    time_accord_edge.append(edge[0:2].tolist())
                    time_accord_edge.append([edge[1], edge[0]])
                self.time_new_edge_dict.update({edge[2]: time_accord_edge})

            self.N = len(self.edge_dict.keys())
            self.beta = self.r * self.N * (self.l - self.w + 1)  # 行走时间约束
            import_net_end = time.time()
            self.io_cost = self.io_cost + (import_net_end - import_net_start)
            logging.info("finish input dataset")
        except Exception as e:
            logging.error("Load dataset error!")
            print(e)

    def batch_update(self):
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
                model.save(self.result_path+'/model/' + str(i) + '_model')
            except Exception as e:
                logging.info("walk is too less to train")
                print(e)

    def random_walk(self, t, walk_num=30):
        """
        静态的时序网络的embedding训练
        :param t: 时间戳
        :param walk_num: 行走数量
        无返回值，向量均存在文件里
        """
        #  做一个加上时间的新边表（下面就开始遍历新边了），对每个时间片的所有边都进行反向的随机游走
        edge_list_at_new_time = np.array(self.time_new_edge_dict.get(t))
        curr_new_edge_arr = np.insert(edge_list_at_new_time, 2, values=t, axis=1)  # 新方法，加入一个时间列

        #  对每个边都以一个batch一个batch地存边
        walks = []
        for each_new_edge in curr_new_edge_arr:
            for i in range(walk_num):
                walk_index = self.reverse_temporal_walk(each_new_edge.tolist())  # 这里简化为l，对结果影响不大
                walk_index.reverse()
                if self.w < len(walk_index) < self.l:
                    walks.append(str(walk_index).replace('[', '').replace(']', ''))

        #  batch处理边
        batch_walks_str = '\n'.join(walks)
        io_start = time.time()
        with open(self.result_path+"/walk/walk.txt", 'a') as f:
            f.write(batch_walks_str)
            f.write('\n')
        io_end = time.time()
        self.io_cost = self.io_cost + io_end - io_start

    def load_walk_set(self):
        """
        把walks读入，并且变成单词的集合格式
        :return: 返回格式符合要求的dataset
        """
        load_walk_start = time.time()
        dataset = []
        with open(self.result_path+"/walk/walk.txt", 'r') as f:
            sourceInLine = f.readlines()
            for line in sourceInLine:
                dataset.append(line.strip('\n').split(', '))
        load_walk_end = time.time()
        self.io_cost = self.io_cost + (float(load_walk_end) - float(load_walk_start))
        return dataset

    def reverse_temporal_walk(self, start_edge):
        """
        调通
        选出一个合法的游走路径
        :return: 从给定节点下的合法游走路径
        """
        curr_walk_index = start_edge[0:2]  # Fetch start_edge[0 and 1]
        curr_edge = start_edge  # 当前的走到的边
        for p in range(1, self.l - 1):  # 注意开闭

            legal_neighbour_edge = self.get_legal_neighbour_edges(curr_edge)  # 传入curr_edge可以相当于传入了时间和位置，每次更新curr_edge即可
            #  需要按照时间次序排个序
            if legal_neighbour_edge != [] and len(legal_neighbour_edge) > 0:
                legal_neighbour_edge.sort(key=lambda x: x[1])
                choose_neighbour = sample_next_edge(curr_edge, legal_neighbour_edge)  # 由于时序不减，所以后面不可能取到这个原来取过的边
                curr_walk_index.append(choose_neighbour[1])  # 把下一个节点写进去
                curr_edge = choose_neighbour  # 因为这个返回的就是一条完整的边，所以可以写这个
            else:
                return curr_walk_index

        return curr_walk_index

    def get_legal_neighbour_edges(self, curr_edge):
        """
        反向游走
        获取一系列的符合要求的edge，以curr_edge的node2为起点，以curr_edge的t为时间约束
        :return legal_neighbour 就是一个二重list的结构，返回的是合法的邻边
        """
        #  注意curr_edge传进来的应当还是完整形态的node1，node2，time
        next_node = curr_edge[1]  # 1是node2的索引
        neighbour = self.edge_dict.get(next_node)  # 是查的出来的，接下来就是取一个合法邻居了
        curr_time = curr_edge[2]
        if neighbour is None:
            return []
        else:
            legal_neighbour = [shift for shift in neighbour if shift[1] < curr_time]
            return legal_neighbour


if __name__ == '__main__':

    usage = "Online CTDNEs params"
    parser = optparse.OptionParser(usage)  # 写入上面定义的帮助信息
    parser.add_option('-r', dest='r', help='Num of walks from each vertex', type='int', default=10)
    parser.add_option('-w', dest='w', help='Lower Bound of the path length', type='int', default=5)
    parser.add_option('-l', dest='l', help='Upper Bound of the path length', type='int', default=80)
    parser.add_option('-d', dest='d', help='Dimension of each embedding vector', type='int', default=128)
    parser.add_option('-g', dest='g', help='Granularity for batch update', type='int', default=10000)
    parser.add_option('-s', dest='s', help='Path of Dataset', type='string', default='foursq2014_TKY_node_format.txt')
    parser.add_option('-j', dest='j', help='Dir of result', type='string', default='result')

    options, args = parser.parse_args()

    #  初始化文件夹
    if not os.path.isdir(options.j):
        os.mkdir(options.j)

    if not os.path.isdir(options.j+"/walk"):
        os.mkdir(options.j+"/walk")

    if not os.path.isdir(options.j+"/model"):
        os.mkdir(options.j+"/model")

    if not os.path.exists(options.j+"/walk/walk.txt"):
        open(options.j+"/walk/walk.txt", 'w').close()

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # 日志格式化输出
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"  # 日期格式
    fp = logging.FileHandler(options.j+"/online_ctdnes_log.log", encoding='gbk')
    fs = logging.StreamHandler()
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT,
                        handlers=[fp, fs])  # 调用handlers=[fp,fs]

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
