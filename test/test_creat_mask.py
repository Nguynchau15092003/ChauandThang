import numpy as np
import unittest
import json


def original_method(max_length, short):
    mask_0 = [[-99999] * max_length for _ in range(max_length)]
    mask_1 = [[-99999] * max_length for _ in range(max_length)]
    mask_2 = [[-99999] * max_length for _ in range(max_length)]
    mask_3 = [[-99999] * max_length for _ in range(max_length)]
    mask_4 = [[-99999] * max_length for _ in range(max_length)]
    mask_5 = [[-99999] * max_length for _ in range(max_length)]
    mask_6 = [[-99999] * max_length for _ in range(max_length)]
    mask_7 = [[-99999] * max_length for _ in range(max_length)]
    mask_8 = [[-99999] * max_length for _ in range(max_length)]
    mask_9 = [[-99999] * max_length for _ in range(max_length)]
    mask_10 = [[-99999] *
               max_length for _ in range(max_length)]
    short_length = len(short)
    assert len(short) == len(short[0])
    for i in range(short_length):
        for j in range(short_length):
            mask_0[i][j] = 0
            if short[i][j] == 1:
                mask_1[i][j] = 0
                mask_2[i][j] = 0
                mask_3[i][j] = 0
                mask_4[i][j] = 0
                mask_5[i][j] = 0
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 2:
                mask_2[i][j] = 0
                mask_3[i][j] = 0
                mask_4[i][j] = 0
                mask_5[i][j] = 0
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 3:
                mask_3[i][j] = 0
                mask_4[i][j] = 0
                mask_5[i][j] = 0
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 4:
                mask_4[i][j] = 0
                mask_5[i][j] = 0
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 5:
                mask_5[i][j] = 0
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 6:
                mask_6[i][j] = 0
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 7:
                mask_7[i][j] = 0
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 8:
                mask_8[i][j] = 0
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 9:
                mask_9[i][j] = 0
                mask_10[i][j] = 0
            elif short[i][j] == 10:
                mask_10[i][j] = 0

    for i in range(short_length):
        mask_1[i][i] = 0
        mask_2[i][i] = 0
        mask_3[i][i] = 0
        mask_4[i][i] = 0
        mask_5[i][i] = 0
        mask_6[i][i] = 0
        mask_7[i][i] = 0
        mask_8[i][i] = 0
        mask_9[i][i] = 0
        mask_10[i][i] = 0

    short_mask = np.asarray(
        [mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8, mask_9, mask_10], dtype='float32')
    return short_mask


def new_method(max_length, short):
    mask = np.full((11, max_length, max_length), -99999, dtype='float32')
    short_length = len(short)
    assert len(short) == len(short[0])
    mask[0, :short_length, :short_length] = 0
    for min_dis in range(1, 11):
        for i in range(short_length):
            for j in range(short_length):
                if short[i][j] >= 1 and short[i][j] <= min_dis:
                    mask[min_dis][i][j] = 0
        for i in range(short_length):
            mask[min_dis][i][i] = 0
    return mask


def original_method_bert(max_length, context_len, short, column_short):
    mask_0 = [[-99999] * max_length for _ in range(max_length)]
    mask_1 = [[-99999] * max_length for _ in range(max_length)]
    mask_2 = [[-99999] * max_length for _ in range(max_length)]
    mask_3 = [[-99999] * max_length for _ in range(max_length)]
    mask_4 = [[-99999] * max_length for _ in range(max_length)]
    mask_5 = [[-99999] * max_length for _ in range(max_length)]
    mask_6 = [[-99999] * max_length for _ in range(max_length)]
    mask_7 = [[-99999] * max_length for _ in range(max_length)]
    mask_8 = [[-99999] * max_length for _ in range(max_length)]
    mask_9 = [[-99999] * max_length for _ in range(max_length)]
    mask_10 = [[-99999] *
               max_length for _ in range(max_length)]
    short_length = len(short)
    assert len(short) == len(short[0])
    for i in range(1):
        for j in range(context_len):
            mask_0[i][j] = 0
            mask_1[i][j] = 0
            mask_2[i][j] = 0
            mask_3[i][j] = 0
            mask_4[i][j] = 0
            mask_5[i][j] = 0
            mask_6[i][j] = 0
            mask_7[i][j] = 0
            mask_8[i][j] = 0
            mask_9[i][j] = 0
            mask_10[i][j] = 0
    for i in range(context_len):
        for j in range(context_len):
            mask_0[i + 1][j + 1] = 0
            if column_short[i][j] == 1:
                mask_1[i + 1][j + 1] = 0
                mask_2[i + 1][j + 1] = 0
                mask_3[i + 1][j + 1] = 0
                mask_4[i + 1][j + 1] = 0
                mask_5[i + 1][j + 1] = 0
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 2:
                mask_2[i + 1][j + 1] = 0
                mask_3[i + 1][j + 1] = 0
                mask_4[i + 1][j + 1] = 0
                mask_5[i + 1][j + 1] = 0
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 3:
                mask_3[i + 1][j + 1] = 0
                mask_4[i + 1][j + 1] = 0
                mask_5[i + 1][j + 1] = 0
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 4:
                mask_4[i + 1][j + 1] = 0
                mask_5[i + 1][j + 1] = 0
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 5:
                mask_5[i + 1][j + 1] = 0
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 6:
                mask_6[i + 1][j + 1] = 0
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 7:
                mask_7[i + 1][j + 1] = 0
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 8:
                mask_8[i + 1][j + 1] = 0
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 9:
                mask_9[i + 1][j + 1] = 0
                mask_10[i + 1][j + 1] = 0
            elif column_short[i][j] == 10:
                mask_10[i + 1][j + 1] = 0
    short_mask = np.asarray(
        [mask_0, mask_1, mask_2, mask_3, mask_4, mask_5, mask_6, mask_7, mask_8, mask_9, mask_10], dtype='float32')
    return short_mask


def new_method_bert(max_length, context_len, short, column_short):
    mask = np.full((11, max_length, max_length), -99999, dtype='float32')
    short_length = len(short)
    assert len(short) == len(short[0])
    mask[:, 0, :context_len] = 0
    mask[0, 1:context_len + 1, 1:context_len + 1] = 0
    for min_dis in range(1, 11):
        for i in range(context_len):
            for j in range(context_len):
                if column_short[i][j] >= 1 and column_short[i][j] <= min_dis:
                    mask[min_dis][i + 1][j + 1] = 0
    return mask


class TestCreatMask(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.max_length = 100
        self.random_short = np.random.randint(1, 10, size=[100, 50, 50])
        with open("dataset/Laptops_corenlp/test_preprocessed.json", 'r') as f:
            self.data = json.load(f)
            f.close()

    def test_create_mask_random(self):
        for id in range(100):
            short = self.random_short[id, :, :]
            short = short - np.diag(np.diag(short))
            orig_res = original_method(self.max_length, short)
            new_res = new_method(self.max_length, short)
            self.assertTrue(np.array_equal(orig_res, new_res))

    def test_create_mask_file(self):
        for obj in self.data:
            short = obj['short']
            orig_res = original_method(self.max_length, short)
            new_res = new_method(self.max_length, short)
            self.assertTrue(np.array_equal(orig_res, new_res))

    def test_create_mask_random_bert(self):
        for id in range(100):
            short = self.random_short[id, :, :]
            short = short - np.diag(np.diag(short))
            orig_res = original_method_bert(
                self.max_length, self.max_length // 10, short, short)
            new_res = new_method_bert(
                self.max_length, self.max_length // 10, short, short)
            self.assertTrue(np.array_equal(orig_res, new_res))

    def test_create_mask_file_bert(self):
        for obj in self.data:
            short = obj['short']
            short = short - np.diag(np.diag(short))
            orig_res = original_method_bert(
                self.max_length, self.max_length // 50, short, short)
            new_res = new_method_bert(
                self.max_length, self.max_length // 50, short, short)
            self.assertTrue(np.array_equal(orig_res, new_res))
