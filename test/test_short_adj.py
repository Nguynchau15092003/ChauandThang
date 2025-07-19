import json
from typing import DefaultDict
import numpy as np
from utils.data_preprocessor import short_adj_generation
import unittest


def get_short_origin(d):
    head = list(d['head'])
    max = len(head)
    tmp = [[0] * max for _ in range(max)]
    for i in range(max):
        j = head[i]
        if j == 0:
            continue
        tmp[i][j - 1] = 1
        tmp[j - 1][i] = 1

    tmp_dict = DefaultDict(list)

    for i in range(max):
        for j in range(max):
            if tmp[i][j] == 1:
                tmp_dict[i].append(j)

    leverl_degree = [[5] * max for _ in range(max)]

    for i in range(max):
        node_set = set()
        leverl_degree[i][i] = 0
        node_set.add(i)
        for j in tmp_dict[i]:
            if j not in node_set:
                leverl_degree[i][j] = 1
                # print(word_leverl_degree)
                node_set.add(j)
            for k in tmp_dict[j]:
                # print(tmp_dict[j])
                if k not in node_set:
                    leverl_degree[i][k] = 2
                    # print(word_leverl_degree)
                    node_set.add(k)
                    for g in tmp_dict[k]:
                        if g not in node_set:
                            leverl_degree[i][g] = 3
                            # print(word_leverl_degree)
                            node_set.add(g)
                            for q in tmp_dict[g]:
                                if q not in node_set:
                                    leverl_degree[i][q] = 4
                                    # print(word_leverl_degree)
                                    node_set.add(q)
    d['short'] = leverl_degree
    return np.array(d['short'])


def get_short_new(d):
    short_adj = short_adj_generation(d['head'])
    return np.array(short_adj)


class TestShortAdj(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        with open("dataset/Laptops_corenlp/train.json", 'r') as f:
            self.data = json.load(f)
            f.close()

    def test_short_adj(self):
        for d in self.data:
            short_origin = get_short_origin(d)
            short_new = get_short_new(d)
            self.assertTrue(np.array_equal(short_origin, short_new))
