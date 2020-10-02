#!python3
# pam.py

import numpy as np


def isint(x):
    return round(x) == x


def uint16_to_bytes(x):
    assert isint(x)
    return int(x).to_bytes(2, 'little')


def uint32_to_bytes(x):
    assert isint(x)
    return int(x).to_bytes(4, 'little')


class PAM(object):
    def __init__(self, name=None, shape=None, slices=None):
        self.name = name
        self.shape = shape
        self.slices = [] if slices is None else slices


class PAMSlice(object):
    footer_bytes = bytes([7] + [0] * 31)
    end_of_features_bytes = bytes([0] * 4)
    separator_bytes = bytes([1, 16, 192, 219])

    def __init__(self, features=None):
        self.features = [] if features is None else features


class PAMFeature(object):

    @staticmethod
    def point_to_bytes(point):
        byte_list = [uint16_to_bytes(x) for x in point]
        return bytes().join(byte_list)


class RectanglePAMFeature(PAMFeature):
    header_bytes = bytes([1, 0, 0, 16])
    separator_bytes = bytes([4, 0, 0, 0, 0, 64, 0, 0])
    footer_bytes = bytes([0, 1, 1, 129])

    def __init__(self, top_left_point=None, size=None):
        self.top_left_point = top_left_point
        self.size = size

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, x):
        self._size = None if x is None else np.array(x)

    @property
    def top_left_point(self):
        return self._top_left_point

    @top_left_point.setter
    def top_left_point(self, x):
        self._top_left_point = None if x is None else np.array(x)

    @property
    def __bytes__(self):
        bytes_list = (
            [self.separator_bytes]
            + [self.point_to_bytes(x) for x in self._all_points]
            + [self.footer_bytes]
        )
        most_bytes = bytes().join(bytes_list)
        display_len = len(most_bytes)
        return self.header_bytes + uint32_to_bytes(display_len) + most_bytes

    @property
    def top_right_point(self):
        return self.top_left_point + (self.size[0] * [1, 0])

    @property
    def bottom_right_point(self):
        return self.top_left_point + self.size

    @property
    def bottom_left_point(self):
        return self.top_left_point + (self.size[1] * [0, 1])

    @property
    def _all_points(self):
        return [self.top_left_point,
                self.top_right_point,
                self.bottom_right_point,
                self.bottom_left_point]



class EmptyPAMFeature(PAMFeature):
    header_bytes = bytes([2, 0, 0, 16])