# -*- coding: UTF-8 -*-

import networkx as nx
import sys
import os
import time
import argparse


# cd F:\dynamic_he\dataset\slide_data_for_link_prediction
# python get_LinkPred_test.py --input ../TKY/full_graph.txt --output-dir TKY190_test_set/ --test-skip 100000
# python get_LinkPred_test.py --input ../Tmall/full_graph.txt --output-dir ./Tmall190_skip50_test_set/ --test-skip 50

# python get_LinkPred_test.py --input ../sig_enron/enron_dele_na_others.txt --output-dir enron190_skip5_test_set/ --test-skip 5


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Random Walk")
    parser.add_argument('--input', nargs='?', default='../dataset/test/phone.txt',
                        help='Input graph path')
    parser.add_argument('--output-dir', nargs='?', default='test/',
                        help='Output file path')

    # 把滑动窗口要用的初始变量替换成滑动窗口的时间长度
    # parser.add_argument('--slide-window-size', type=float, default=100000)
    parser.add_argument('--test-skip', type=int, default=10000)

    return parser.parse_args()


# 把动态图以时间为索引，重新进行组织存在time2data（所有时间片）
def load_graph(input_graph):
    time_list = set()
    time2data = dict()
    with open(input_graph) as f:
        for line in f:
            # toks = line.strip().split('\t')
            toks = line.strip().split('\t')
            t = float(toks[4])
            if t not in time2data:
                time2data[t] = []
                time_list.add(t)
            time2data[t].append(line)
    if time_list.__len__() == 0:
        print("The time list is empty.")
        exit()
    time_list = sorted(list(time_list))
    print("time list len:", time_list.__len__())
    return time_list, time2data


def write_slide_data(filename, t_old, t_new, timelist, time2data_dict):
    f = open(filename, 'w')
    for ti in timelist:
        if ti <= t_old:
            continue
        if ti > t_new:
            break  # timelist是已经排好序的时间列表
        for line in time2data_dict[ti]:
            toks = line.strip().split(',')
            if toks.__len__() > 1:
                f.write('\t'.join(toks) + '\n')
            else:
                f.write(line)


def main(args):
    # assign output path
    output_dir_slide_data = args.output_dir
    if not os.path.exists(output_dir_slide_data):
        os.makedirs(output_dir_slide_data)

    # load graph
    time_list, time2data = load_graph(args.input)
    print("loaded graph\n")

    # 循环进行滑动窗口
    # 对于给定时间序列中的每n_skip个时间循环一次（时间序列由load graph读文件给出）
    n_skip = args.test_skip
    # slide_window_size = args.slide_window_size
    t_num = 0
    t_new = time_list[0] - 1
    for ti in time_list:
        t_num += 1
        # 每次窗口向前滑动n_skip个时间片
        if (t_num % n_skip != 0) and t_num < time_list.__len__():
            continue

        # 有效时间区间: (t_old,t_new]
        t_old = t_new
        t_new = ti

        # 有效数据写文件
        write_slide_data(output_dir_slide_data + str(t_num) + '.txt', t_old, t_new, time_list, time2data)


if __name__ == '__main__':
    args = parse_args()
    startt = time.process_time()
    main(args)
    endt = time.process_time()
    print("Run time:", endt - startt)
