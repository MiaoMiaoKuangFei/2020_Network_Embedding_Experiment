import numpy as np
import random
import time


def load_metapath(input_file, t1, t2, seq):
    start = time.time()
    t1_t2 = dict()
    t2_t1 = dict()
    edges = np.loadtxt(input_file, delimiter="\t").astype(np.int).astype(str)
    times = np.sort(edges[:, 4].astype(np.int))
    threshold = times[-1 if seq >= len(times) else seq]
    print(len(times))
    print(len(set(times)))
    print("threshold:", threshold)
    print(len(times))
    for edge in edges:
        if edge[1] == edge[3] or edge[1] not in [t1, t2] or edge[3] not in [t1, t2] or int(edge[4]) > threshold:
            continue

        if (edge[1] == t1 and edge[3] == t2) or (edge[1] == t2 and edge[3] == t1):
            vertex_t1 = edge[0] if edge[1] == t1 else edge[2]
            vertex_t2 = edge[0] if edge[1] == t2 else edge[2]
            if t1_t2.get(vertex_t1) is not None:
                #  为什么不能update{append}，因为append返回值为None
                t1_t2.get(vertex_t1).append(vertex_t2)
            else:
                t1_t2.update({vertex_t1: [vertex_t2]})

            if t2_t1.get(vertex_t2) is not None:
                t2_t1.get(vertex_t2).append(vertex_t1)
            else:
                t2_t1.update({vertex_t2: [vertex_t1]})
    end = time.time()
    return t1_t2, t2_t1, end - start


class MetaPathGenerator:
    def __init__(self, t1_t2_list, t2_t1_list):
        self.t1_t2_list = t1_t2_list
        self.t2_t1_list = t2_t1_list

    def generate_random_212(self, outfile_name, num_walks, walk_length):
        io_time = 0
        outfile = open(outfile_name, 'a')
        print(len(self.t2_t1_list) * num_walks)
        for peri in self.t2_t1_list:
            peri0 = peri
            for j in range(0, num_walks):  # wnum walks
                outline = peri0
                for i in range(0, walk_length):
                    t1s = self.t2_t1_list[peri]
                    numc = len(t1s)
                    t1id = random.randrange(numc)
                    t1 = t1s[t1id]
                    outline += "," + t1
                    peris = self.t1_t2_list[t1]
                    nump = len(peris)
                    periid = random.randrange(nump)
                    peri = peris[periid]
                    outline += "," + peri
                start = time.time()
                outfile.write(outline + "\n")
                end = time.time()
                io_time += end - start

        outfile.close()
        return io_time

    def generate_random_121(self, outfile_name, num_walks, walk_length):
        io_time = 0
        outfile = open(outfile_name, 'a')
        print(len(self.t1_t2_list) * num_walks)
        for peri in self.t1_t2_list:
            peri0 = peri
            for j in range(0, num_walks):  # wnum walks
                outline = peri0
                for i in range(0, walk_length):
                    t2s = self.t1_t2_list[peri]
                    numc = len(t2s)
                    t2_id = random.randrange(numc)
                    t2 = t2s[t2_id]
                    outline += "," + t2
                    peris = self.t2_t1_list[t2]
                    nump = len(peris)
                    periid = random.randrange(nump)
                    peri = peris[periid]
                    outline += "," + peri
                start = time.time()
                outfile.write(outline + "\n")
                end = time.time()
                io_time += end - start

        outfile.close()
        return io_time
