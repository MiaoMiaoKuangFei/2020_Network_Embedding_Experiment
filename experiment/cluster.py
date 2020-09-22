# @desc   : 完成点分类任务
import numpy as np
import logging
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn import metrics as mr
import util


class node_cluster:
    def __init__(self):
        logging.info("初始化聚类器")
        self.vectors_dict = {}  # 每个点对应的字典{点index(str):向量(nparray)} 这个str可不能带小数点
        self.node_dict = {}  # 每个点的label
        self.node_vec = []  # 备选点
        self.node_label = []

    # 操作：首先导入数据集，格式不变；原来的test_set输入为以空格为分隔符的node1 node2，主程序的embedding，格式不变

    def import_model(self, model_name):
        #  0代表主方法，model_name代表的是主函数输出的节点向量
        self.vectors_dict = util.import_model(model_name)

    def import_node(self, dsname):
        # 统一操作为str
        self.node_dict = util.import_node_label_dict(dsname)

        for node in self.node_dict:
            if self.vectors_dict.get(str(node)) is not None:
                self.node_vec.append(self.vectors_dict.get(node))
                self.node_label.append(self.node_dict.get(node))

    def cluster(self):

        estimator = KMeans(n_clusters=2)  # 构造聚类器
        estimator.fit(self.node_vec)  # 聚类
        label_predict = estimator.predict(self.node_vec)  # 获取聚类标签

        print(mr.normalized_mutual_info_score(label_predict, self.node_label))


if __name__ == '__main__':
    util.log_def()
    options, args = util.model_choice_dataset_args("cluster")

    nc = node_cluster()
    nc.import_model(options.m)
    nc.import_node(options.d)
    nc.cluster()
