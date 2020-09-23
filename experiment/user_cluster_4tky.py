import pandas as pd
import util
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

if __name__ == '__main__':
    #  若是找不到文件，那么要设定工作路径在项目路径上
    options, args = util.model_choice_dataset_args("user_cluster")

    c = util.parse_model_name(options.m)

    vertex2vec = util.import_model(options.m)

    nodeId_2true_labelId = pd.read_csv("experiment\\dataset\\nodeID_labelID_tuple.csv")
    label_true_list = list(nodeId_2true_labelId.iloc[:, 1])
    label_true_map = {}
    for tup in nodeId_2true_labelId.values:
        label_true_map.update({int(tup[0]): tup[1]})

    edge_dict, time_dict, io_cost = util.import_net(options.d)
    user_id = []
    user_type = []
    for edge in edge_dict:
        if edge not in label_true_map.keys():
            cnt_type = np.ones(9)
            for pos_vertex in edge_dict.get(edge):
                cnt_type[label_true_map.get(pos_vertex[0])] += 1
            user_id.append(edge)
            user_type.append(np.argmax(cnt_type))
    vec = []
    label = []
    for i, uid in enumerate(user_id):
        if vertex2vec.get(str(uid)) is not None:
            vec.append(vertex2vec.get(str(uid)).tolist())
            label.append(user_type[i])
    #  至此，就有了vec为user向量，user_type作为标签
    estimator = KMeans(n_clusters=9)  # 构造聚类器
    estimator.fit(vec)  # 聚类
    label_predict = estimator.predict(vec)  # 获取聚类标签

    print(nmi(label, label_predict))
