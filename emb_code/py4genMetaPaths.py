import sys
import os
import random
from collections import Counter
import time


class MetaPathGenerator:
	def __init__(self, t1_t2_list, t2_t1_list):
		# core -> t1 , peri -> t2
		# self.t1_t2_list = dict()
		# self.t2_t1_list = dict()
		self.t1_t2_list = t1_t2_list
		self.t2_t1_list = t2_t1_list

	def generate_random_212(self, outfilename, numwalks, walklength):
		outfile = open(outfilename, 'a')
		for peri in self.t2_t1_list:
			peri0 = peri
			for j in range(0, numwalks):  # wnum walks
				outline = peri0
				for i in range(0, walklength):
					t1s = self.t2_t1_list[peri]
					numc = len(t1s)
					t1id = random.randrange(numc)
					t1 = t1s[t1id]
					outline += " " + t1
					peris = self.t1_t2_list[t1]
					nump = len(peris)
					periid = random.randrange(nump)
					peri = peris[periid]
					outline += " " + peri
				outfile.write(outline + "\n")
		outfile.close()

	def generate_random_121(self, outfile_name, num_walks, walk_length):
		io_time = 0
		outfile = open(outfile_name, 'a')  # 修改为append！
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
					outline += " " + t2
					peris = self.t2_t1_list[t2]
					nump = len(peris)
					periid = random.randrange(nump)
					peri = peris[periid]
					outline += " " + peri
				start = time.time()
				outfile.write(outline + "\n")
				end = time.time()
				io_time += end - start

		outfile.close()
		return io_time

