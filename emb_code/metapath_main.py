from gensim.models import Word2Vec

import metapath_generator
import util
import os
import optparse
import time
import logging

if __name__ == '__main__':

    usage = "Metapath-Method params"
    parser = optparse.OptionParser(usage)
    parser.add_option('-r', dest='r', help='Num of walks from each vertex', type='int', default=10)
    parser.add_option('-l', dest='l', help='Path length', type='int', default=80)
    parser.add_option('-d', dest='d', help='Dimension of each embedding vector', type='int', default=128)
    parser.add_option('-s', dest='s', help='Path of Dataset', type='string', default='foursq2014_TKY_node_format.txt')
    parser.add_option('-j', dest='j', help='Dir of result', type='string', default='result')
    parser.add_option('-c', dest='c', help='Dataset', type='string', default='tky')
    parser.add_option('-q', dest='q', help='Time seq', type='int', default=100000)
    options, args = parser.parse_args()
    c = options.c  # tky,enron
    #  initialize path and dir and file

    total_time_dict = {}
    io_time_dict = {}
    minus_io_time_dict = {}

    for each_i in [130000, 140000, 150000, 160000]:
        options.j = 'result/' + str(each_i) + '_result'
        options.q = each_i
        if not os.path.isdir(options.j):
            os.mkdir(options.j)

        if not os.path.isdir(options.j + "/walk"):
            os.mkdir(options.j + "/walk")

        if not os.path.isdir(options.j + "/model"):
            os.mkdir(options.j + "/model")

        if not os.path.exists(options.j + "/walk/walk.txt"):
            open(options.j + "/walk/walk.txt", 'w').close()

        util.log_def(options.j + "/log.log")
        if c == 'tky':
            t_1_t_2_tuples_list = [('1', '0')]
        else:
            # t_1_t_2_tuples_list = [('1', '2'), ('2', '1'), ('4', '10'), ('5', '6'), ('6', '9'), ('7', '8'), ('8', '7'),
            # ('9', '6'), ('10', '4')]
            t_1_t_2_tuples_list = [('1', '2'), ('2', '1'), ('4', '5'), ('5', '4'), ('6', '7'), ('7', '6'), ('8', '9'),
                                   ('9', '8'), ('10', '1')]
        io_time = 0
        total_time = 0
        start = time.time()
        for tu in t_1_t_2_tuples_list:
            t1_t2, t2_t1, io_1 = metapath_generator.load_metapath(options.s, tu[0], tu[1], options.q)
            mt = metapath_generator.MetaPathGenerator(t1_t2, t2_t1)
            io_2 = 0  # mt.generate_random_212(options.j + "/walk/walk.txt", options.r, options.l)
            io_3 = mt.generate_random_121(options.j + "/walk/walk.txt", options.r, options.l)
            io_time = io_time + io_1 + io_2 + io_3
        load_walk_start = time.time()
        dataset = []
        with open(options.j + "/walk/walk.txt", 'r') as f:
            sourceInLine = f.readlines()
            for line in sourceInLine:
                dataset.append(line.strip('\n').split(','))
        load_walk_end = time.time()
        io_time += (load_walk_end - load_walk_start)
        model = Word2Vec(dataset, sg=1, size=options.d, window=10, min_count=0, workers=8)
        # model.save(options.j + '/model/metapath_model')
        #  训练也要写在里面
        end = time.time()
        logging.info("Total time(include I/O time loss)" + str(end - start))
        logging.info("I/O time loss" + str(io_time))
        logging.info("Total time(exclude I/O time loss)" + str(end - start - io_time))

        total_time_dict.update({each_i: end - start})
        io_time_dict.update({each_i: io_time})
        minus_io_time_dict.update({each_i: end - start - io_time})
