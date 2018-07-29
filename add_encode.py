# coding:utf-8

import numpy as np


class Table(object):

    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, s, row_number):
        x = np.zeros((row_number, len(self.chars)))
        for i, c in enumerate(s):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = np.argmax(x, axis=-1)
        return ''.join(self.indices_char[x] for x in x)
