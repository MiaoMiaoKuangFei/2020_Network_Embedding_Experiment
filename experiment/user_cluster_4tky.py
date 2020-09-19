import pandas as pd
import util

if __name__ == '__main__':
    #  若是找不到文件，那么要设定工作路径在项目路径上
    options, args = util.model_choice_dataset_args("user_cluster")

    c = util.parse_model_name(options.m)

    vertex2vec = util.import_model(options.m)

    nodeId_2true_labelId = pd.read_csv("experiment\\dataset\\nodeID_labelID_tuple.csv")
    label_true_list = list(nodeId_2true_labelId.iloc[:, 1])
    label_true_map = {}
    for tup in nodeId_2true_labelId.values:
        label_true_map.update({str(tup[0]): tup[1]})

    edge_dict, time_dict, io_cost = util.import_net(options.d)
