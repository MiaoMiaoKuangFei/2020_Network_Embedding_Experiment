"""
单点聚类实验
nodeID_labelID_tuple.csv是正例
"""
from gensim.models import Word2Vec
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
import numpy as np
import util

if __name__ == '__main__':
    options, args = util.model_choice_dataset_args("single_cluster")
    op = util.parse_model_name(options.m)
    util.log_def()

    #  这里是节点id->标签id（已经是标记好的二级标签）
    nodeId_2true_labelId = pd.read_csv("experiment\\dataset\\nodeID_labelID_tuple.csv")
    label_true_list = list(nodeId_2true_labelId.iloc[:, 1])
    label_true_map = {}
    for tup in nodeId_2true_labelId.values:
        label_true_map.update({str(tup[0]): tup[1]})

    #  正确标签列表（如果没有vec的则不放在里面）
    label_true_except_no_vec = []
    #  向量列表
    vec_list = []

    #  导入训练好的模型
    if op == 1:
        model = Word2Vec.load(options.m)
        for i in range(len(nodeId_2true_labelId)):
            if i % 10000 == 0:
                print(i, "轮，总共", len(nodeId_2true_labelId), "轮")
            try:
                vec_list.append(model.wv.__getitem__(str(nodeId_2true_labelId.iloc[i, 0])))
                label_true_except_no_vec.append(label_true_list[i])
            except:
                continue
    else:
        word_vectors = np.loadtxt(options.m, delimiter=' ')
        for e, line in enumerate(word_vectors):
            if e == 50:
                print(label_true_except_no_vec)
            foo = label_true_map.get(str(int(line[0])))
            if foo is not None:
                label_true_except_no_vec.append(foo)
                vec_list.append(line[1:-1])

    print("删除没有vec的点之后，正例数量为", len(label_true_except_no_vec))
    print("获取的vec列表长度为", len(vec_list))
    #  一共分标签到9类
    estimator = KMeans(n_clusters=9)  # 构造聚类器
    estimator.fit(vec_list)  # 聚类
    label_predict = estimator.predict(vec_list)  # 获取聚类标签

    print(nmi(label_true_except_no_vec, label_predict))
