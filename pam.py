#!python3
# pam.py

import numpy as np
import os


default_encoding = 'utf8'
default_byteorder = 'little'  # ... endian


def isint(x):
    return round(x) == x


def iseven(x):
    return (x % 2) == 0


def uint_to_bytes(x,
                  length,
                  byteorder=default_byteorder,
                  allow_overflow=False):
    assert isint(x)  # TODO - add message
    assert x >= 0  # TODO - add message

    x_prime = x % ((2 ** 8) ** length) if allow_overflow else x

    return int(x_prime).to_bytes(length, byteorder=byteorder, signed=False)


def uint8_to_bytes(x, **kwargs):
    return uint_to_bytes(x, length=1, **kwargs)


def uint16_to_bytes(x, **kwargs):
    return uint_to_bytes(x, length=2, **kwargs)


def uint32_to_bytes(x, **kwargs):
    return uint_to_bytes(x, length=4, **kwargs)


def string_to_bytes(s, encoding=default_encoding):
    return bytes(s, encoding=encoding)


class PAM(object):
    def __init__(self, name=None, shape=None, slices=None):
        self.name = name
        self.shape = shape
        self.slices = [] if slices is None else slices

    def __bytes__(self):
        name_bytes = string_to_bytes(self.name)
        name_len_bytes = uint8_to_bytes(len(name_bytes))
        bytes_list = ([name_len_bytes, name_bytes]
                      + [uint32_to_bytes(x) for x in self.shape]
                      + [bytes(s) for s in self.slices])
        return bytes().join(bytes_list)

    def __repr__(self):
        pass  # TODO

    def write_binary(self, filepath):
        with open(filepath, 'wb') as fle:
            fle.write(bytes(self))

    def check_consistency(self):
        pass  # TODO


class PAMSlice(object):
    footer_bytes = bytes([7] + [0] * 31)
    end_of_features_bytes = bytes([0] * 4)
    separator_bytes = bytes([1, 16, 192, 219])

    def __init__(self, features=None, unknown_order=0):
        self.features = [] if features is None else features
        self.unknown_order = unknown_order

    def __bytes__(self):
        tail_bytes_list = (
            [self._unknown_bytes,
             self.separator_bytes,
             self._counting_separator_bytes]
            + [bytes(f) for f in self.features]
            + [self.end_of_features_bytes])
        tail_bytes = bytes().join(tail_bytes_list)
        tail_len = len(tail_bytes)
        tail_len_bytes = uint32_to_bytes(tail_len)
        tail_len_minus_eight_bytes = uint32_to_bytes(tail_len - 8)
        return (tail_len_bytes + tail_len_minus_eight_bytes
                + tail_bytes + self.footer_bytes)

    def __repr__(self):
        pass  # TODO

    @property
    def unknown_order(self):
        return self._unknown_order

    @unknown_order.setter
    def unknown_order(self, x):
        assert isint(x)  # TODO - add message
        self._unknown_order = x

    @property
    def _unknown_bytes(self):
        unknown_bytes = None

        if self._isempty:
            unknown_bytes = bytes([14, 129, 0, 108])
        elif len(self.features) == 1:
            feature_type = type(self.features[0])
            if feature_type is InvertPAMFeature:
                unknown_bytes = bytes([98, 250, 255, 121])

        if unknown_bytes is None:
            unknown_bytes = bytes([0] * 4)

        return unknown_bytes

    @property
    def _isempty(self):
        if self.features:
            isempty = (len(self.features) == 1
                       and type(self.features[0]) is EmptyPAMFeature)
        else:
            isempty = None
        return isempty

    @property
    def _counting_separator_bytes(self):
        order = self.unknown_order
        return uint32_to_bytes(order * 2) + (uint32_to_bytes(2) * order)


class PAMFeature(object):
    header_bytes = None
    tail_bytes = None

    def __init__(self, header_bytes=None, tail_bytes=None):
        if header_bytes is not None:
            self.header_bytes = header_bytes

        if tail_bytes is not None:
            self.tail_bytes = tail_bytes

    def __bytes__(self):
        return self.header_bytes + self.tail_bytes

    @staticmethod
    def point_to_bytes(point):
        byte_list = [uint16_to_bytes(x) for x in point]
        return bytes().join(byte_list)


class RectanglePAMFeature(PAMFeature):
    header_bytes = bytes([1, 0, 0, 16])
    separator_bytes = bytes([4, 0, 0, 0,
                             0, 64, 0, 0])
    footer_bytes = bytes([0, 1, 1, 129])

    def __init__(self, top_left_point=None, size=None):
        super().__init__()
        self.top_left_point = top_left_point
        self.size = size

    def __bytes__(self):
        tail_bytes = self.tail_bytes
        tail_len_bytes = uint32_to_bytes(len(tail_bytes))
        return self.header_bytes + tail_len_bytes + tail_bytes

    def __repr__(self):
        top_left_point_repr = (None if self.top_left_point is None else
                               self.top_left_point.tolist())
        size_repr = (None if self.size is None else
                     self.size.tolist())
        return '''{:s}(top_left_point={}, size={})'''.format(
            self.__class__.__name__,
            top_left_point_repr,
            size_repr)

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

    @property
    def tail_bytes(self):
        tail_bytes_list = (
                [PAMSlice.separator_bytes,
                 self.separator_bytes]
                + [self.point_to_bytes(x) for x in self._all_points]
                + [self.footer_bytes])
        return bytes().join(tail_bytes_list)


class PointPAMFeature(RectanglePAMFeature):
    def __init__(self, point=None):
        super().__init__(top_left_point=point, size=[1, 1])

    def __repr__(self):
        point_repr = (None if self.point is None else
                      self.point.tolist())
        return '''{:s}(point={})'''.format(self.__class__.__name__, point_repr)

    @property
    def point(self):
        return self.top_left_point

    @point.setter
    def point(self, x):
        self.top_left_point = x


class EmptyPAMFeature(PAMFeature):
    header_bytes = bytes([2, 0, 0, 16])
    tail_bytes = bytes()

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class InvertPAMFeature(PAMFeature):
    header_bytes = bytes([0, 0, 0, 16])
    tail_bytes = bytes(([0] * 8) + ([0, 0, 0, 67] * 2))

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return self.__class__.__name__ + '()'


def main():
    pam = PAM(name='cool_pam', shape=(64, 64, 2))
    pam.slices = [PAMSlice() for _ in range(pam.shape[2])]
    pam.slices[0].features.append(EmptyPAMFeature())
    pam.slices[1].features.append(PointPAMFeature(point=(1, 1)))

    print(np.array([b for b in bytes(pam)]))

    filepath = os.path.join('data', 'gen_test.pam')
    pam.write_binary(filepath)


if __name__ == '__main__':
    main()
