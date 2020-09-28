import argparse
import time
import networkx
import os
from py4genMetaPaths import MetaPathGenerator
from gensim.models import word2vec


# cd F:\dynamic_he\baseline\change2vec_gjw
# python change2vec.py --input-dir ../../dataset/slide_data_for_link_prediction/TKY190_full/ --output-dir ../output/change2vec_gjw/TKY190_full/
# python change2vec.py --input-dir ../../dataset/slide_data_for_link_prediction/enron190_skip5_full/ --output-dir ../output/change2vec_gjw/enron190_skip5_full/
# 注意在这个方法中使用了metapath，会对节点id之前加字符
# 无向快照(单个快照中无需考虑时间戳)


def parse_args():
    parser = argparse.ArgumentParser(description="change2vec")
    # parser.add_argument('--input-old', nargs='?', default='../../dataset/foursq/change/TKY_change1.txt',
    #                     help='Input old graph path')
    # parser.add_argument('--input-new', nargs='?', default='../../dataset/foursq/change/TKY_change2.txt',
    #                     help='Input new graph path')
    #
    # parser.add_argument('--output-file', nargs='?', default='output_test.txt',
    #                     help='Input new graph path')

    parser.add_argument('--input-dir', nargs='?', default='../../dataset/foursq/change/',
                        help='Input dir of graph')
    parser.add_argument('--output-dir', nargs='?', default='output_rm/',
                        help='Output file path')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    return parser.parse_args()


def load_graph(g_path, g):
    # todo: 由于后面要使用metapath，这里应该对输入的节点id进行类型的头部标记
    with open(g_path) as f:
        for line in f:
            toks = line.strip().split('\t')
            # toks = line.strip().split(',')
            g.add_edge('v' + toks[0], 'a' + toks[2])
    print(networkx.info(g), '\n')


# fixme:把这里的vi[0]修改成访问对应的节点类别应该就能够正常运行了，但是注意节点相关的id可能也需要进行调整，或者不调整但是不利用这个id的首位做判断等事情。
def get_echange(g_old, g_new, vchange, t1_header='v', t2_header='a'):
    t1_t2_list = dict()
    t2_t1_list = dict()

    for vi in vchange:
        if vi in g_new.nodes:
            if vi[0] == t1_header:
                if vi not in t1_t2_list:
                    t1_t2_list[vi] = []
                for i in g_new.neighbors(vi):
                    if i in vchange:
                        t1_t2_list[vi].append(i)
                if vi not in g_old:
                    continue
                for i in g_old.neighbors(vi):
                    if i in g_new.neighbors(vi):
                        continue
                    if i in vchange:
                        t1_t2_list[vi].append(i)
                # t1_t2_list[vi] = list(g_new.neighbors(vi))
            elif vi[0] == t2_header:
                if vi not in t2_t1_list:
                    t2_t1_list[vi] = []
                for i in g_new.neighbors(vi):
                    if i in vchange:
                        t2_t1_list[vi].append(i)
                if vi not in g_old:
                    continue
                for i in g_old.neighbors(vi):
                    if i in g_new.neighbors(vi):
                        continue
                    if i in vchange:
                        t2_t1_list[vi].append(i)
                # t2_t1_list[vi] = list(g_new.neighbors(vi))
        elif vi in g_old.nodes:
            if vi[0] == t1_header:
                if vi not in t1_t2_list:
                    t1_t2_list[vi] = []
                for i in g_old.neighbors(vi):
                    if i in vchange:
                        t1_t2_list[vi].append(i)
                if vi not in g_new:
                    continue
                for i in g_new.neighbors(vi):
                    if i in g_old.neighbors(vi):
                        continue
                    if i in vchange:
                        t1_t2_list[vi].append(i)
                # t1_t2_list[vi] = list(g_old.neighbors(vi))
            elif vi[0] == t2_header:
                if vi not in t2_t1_list:
                    t2_t1_list[vi] = []
                for i in g_old.neighbors(vi):
                    if i in vchange:
                        t2_t1_list[vi].append(i)
                if vi not in g_new:
                    continue
                for i in g_new.neighbors(vi):
                    if i in g_old.neighbors(vi):
                        continue
                    if i in vchange:
                        t2_t1_list[vi].append(i)
                # t2_t1_list[vi] = list(g_old.neighbors(vi))
        else:
            print("The changed node didn't appear in either of the two graph.")
            break

    return t1_t2_list, t2_t1_list


def get_vchange(g_old, g_new):
    vchange = set()
    v_old = set(g_old.nodes)
    v_new = set(g_new.nodes)
    e_old = set(g_old.edges)
    e_new = set(g_new.edges)

    # newly-added nodes & their one-hop neighbor nodes
    v_newly_added = v_new - v_old
    vchange |= v_newly_added
    for i in v_newly_added:
        vchange |= set(g_new.neighbors(i))
    # print(vchange.__len__())

    # deleted nodes & their one-hop neighbor nodes
    v_deleted = v_old - v_new
    vchange |= v_deleted
    for i in v_deleted:
        vchange |= set(g_old.neighbors(i))
    # print(vchange.__len__())

    # newly-formed edges which caused triad closure processes
    e_newly_added = e_new - e_old
    v_cause_closure = set()
    while e_newly_added.__len__() != 0:
        e = e_newly_added.pop()
        en0 = set(g_new.neighbors(e[0]))
        en1 = set(g_new.neighbors(e[1]))
        if (en0 & en1).__len__() != 0:
            v_cause_closure.add(e[0])
            v_cause_closure.add(e[1])
    vchange |= v_cause_closure
    print("v_cause_closure.__len__():", v_cause_closure.__len__())
    # print("e_newly_added.__len__():", e_newly_added.__len__())

    # deleted edges which caused triad open processes
    e_deleted = e_old - e_new
    v_cause_open = set()
    while e_deleted.__len__() != 0:
        e = e_deleted.pop()
        en0 = set(g_old.neighbors(e[0]))
        en1 = set(g_old.neighbors(e[1]))
        if (en0 & en1).__len__() != 0:
            v_cause_open.add(e[0])
            v_cause_open.add(e[1])
    vchange |= v_cause_open
    print("v_cause_open.__len__():", v_cause_open.__len__())
    # print("e_deleted.__len__():", e_deleted.__len__())

    return vchange


def main(args, input_old, input_new, output_dir_rm, output_dir_emb):
    # input_old = os.path.join(args.input_dir, input_old)
    # input_new = os.path.join(args.input_dir, input_new)
    output_file_rm = os.path.join(output_dir_rm, input_old[:-4] + '_' + input_new[:-4] + '.txt')
    output_file_emb = os.path.join(output_dir_emb, input_old[:-4] + '_' + input_new[:-4] + '.emb')

    # 不应该也不需要使用MultiGraph进行构建，否则引起：1.判断引起开闭的边的策略有误，2.原方法中应当是以无重复边实现的
    g_old = networkx.Graph(name='g_old')
    g_new = networkx.Graph(name='g_new')
    load_graph(os.path.join(args.input_dir, input_old), g_old)
    load_graph(os.path.join(args.input_dir, input_new), g_new)

    # 找到changed节点: 4 steps
    v_changed = get_vchange(g_old, g_new)
    print(v_changed.__len__())

    # 提取子图
    metapath_config = [('v', 'a')]  # 往里面加入需要的metapath就行

    for tup in metapath_config:
        #  思路：每一次对于不同的metapath，都提取一次t1t2list和t2t1list，在这一类的metapath基础上游走，然后在generate_random函数里都是以a的形式写walks到文件中
        #  保证每一类metapath行走出来的结果都存到了同一个txt中以供运算
        t1_t2_list, t2_t1_list = get_echange(g_old, g_new, v_changed, tup[0], tup[1])
        mpg = MetaPathGenerator(t1_t2_list, t2_t1_list)

        # 游走(游走结果写文件)
        mpg.generate_random_212(output_file_rm, args.num_walks, args.walk_length)
        mpg.generate_random_121(output_file_rm, args.num_walks, args.walk_length)  # 新增

    print("random walk end\n")

    # 读游走序列文件，输入skipgram生成节点向量。
    sentences = word2vec.LineSentence(output_file_rm)
    model = word2vec.Word2Vec(sentences, size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                              workers=args.workers)
    model.wv.save_word2vec_format(output_file_emb)


# def main(input_old, input_new, output_file, num_walks, walk_length):
#     # 不应该也不需要使用MultiGraph进行构建，否则引起：1.判断引起开闭的边的策略有误，2.原方法中应当是以无重复边实现的
#     g_old = networkx.Graph(name='g_old')
#     g_new = networkx.Graph(name='g_new')
#     load_graph(input_old, g_old)
#     load_graph(input_new, g_new)
#
#     # 找到changed节点: 4 steps
#     v_changed = get_vchange(g_old, g_new)
#     print(v_changed.__len__())
#
#     # 提取子图
#     t1_t2_list, t2_t1_list = get_echange(g_old, g_new, v_changed)
#     mpg = MetaPathGenerator(t1_t2_list, t2_t1_list)
#
#     # 游走(游走结果写文件)
#     mpg.generate_random_212(output_file, num_walks, walk_length)
#     print("random walk end")
#
#     # 读游走序列文件，输入skipgram生成节点向量。
#     sentences = word2vec.LineSentence(output_file)
#     model = word2vec.Word2Vec(sentences, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
#     model.save_word2vec_format(output_file[:-4]+".emb")


if __name__ == '__main__':
    args = parse_args()
    # assign output path
    output_dir_rm = args.output_dir + 'rm/'
    output_dir_emb = args.output_dir + 'emb/'
    if not os.path.exists(output_dir_rm):
        os.makedirs(output_dir_rm)
    if not os.path.exists(output_dir_emb):
        os.makedirs(output_dir_emb)

    tstart = time.process_time()

    files = os.listdir(args.input_dir)
    # files.sort()  # 按字符串排序
    files.sort(key=lambda x: int(x[:-4]))  # 文件名按整数排序
    print('num of files:', os.listdir(args.input_dir).__len__())
    print(files)
    cnt_i = 0
    input_old = ''
    input_new = ''
    # output_file = ''
    for fi in files:
        if os.path.isfile(os.path.join(args.input_dir, fi)):
            cnt_i += 1
            if cnt_i == 1:
                input_old = fi
                input_new = fi
                continue
            input_old = input_new
            input_new = fi
            # output_file = input_old[:-4]+'_'+input_new[:-4]+'.txt'

            # main(os.path.join(args.input_dir, input_old), os.path.join(args.input_dir, input_new),
            #      os.path.join(args.output_dir, output_file), args.num_walks, args.walk_length)
            main(args, input_old, input_new, output_dir_rm, output_dir_emb)

    tend = time.process_time()
    print("total time: ", tend - tstart)
