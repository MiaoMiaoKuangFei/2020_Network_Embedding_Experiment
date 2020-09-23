import pandas as pd
import numpy as np

if __name__ == '__main__':
    input_file = 'experiment/dataset/enron190_skip5_full/20.txt'
    edges = np.loadtxt(input_file, delimiter="\t").astype(np.int)
    types = dict()
    for edge in edges:
        cast_str = str(edge[1]) + ',' + str(edge[3]) if edge[1] < edge[3] else str(edge[3]) + ',' + str(edge[1])

        if cast_str in types:
            types.update({cast_str: types.get(cast_str) + 1})
        else:
            types.update({cast_str: 1})
    with open('cnt_types.txt','w') as f:
        for t in types:
            f.write('('+str(t)+')数量为'+str(types.get(t))+'个\n')

