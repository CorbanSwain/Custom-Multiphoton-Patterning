#!python3
# pam.py

import numpy as np
import os
import enum


default_encoding = 'utf8'
default_byteorder = 'little'  # ... endian
byte_word_length = 4


def isint(x):
    return round(x) == x


def iseven(x):
    return (x % 2) == 0


# to bytes

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


# pop bytes

def pop_byte(byte_arr):
    return byte_arr.pop(0)


def pop_bytes(byte_arr, n):
    return bytes([pop_byte(byte_arr) for _ in range(n)])


def pop_word(byte_arr):
    return pop_bytes(byte_arr, byte_word_length)


def bytes_to_uint(byts, byteorder=default_byteorder):
    return int.from_bytes(byts, byteorder=byteorder, signed=False)


def pop_uint8(byte_arr):
    return pop_byte(byte_arr)


def pop_uint16(byte_arr):
    return bytes_to_uint(pop_bytes(byte_arr, 2))


def pop_uint32(byte_arr):
    return bytes_to_uint(pop_bytes(byte_arr, 4))


def pop_string(byts, n, encoding=default_encoding):
    return pop_bytes(byts, n).decode(encoding)


# peek bytes

def peek_byte(byts):
    return byts[0]


def peek_bytes(byts, n):
    return bytes(byts[:n])


def peek_word(byts):
    return peek_bytes(byts, byte_word_length)


def peek_uint8(byts):
    return peek_byte(byts)


def peek_uint16(byts):
    return bytes_to_uint(peek_bytes(byts, 2))


def peek_uint32(byts):
    return bytes_to_uint(peek_bytes(byts, 4))


def peek_string(byts, n, encoding=default_encoding):
    return peek_bytes(byts, n).decode(encoding)


# class definitions

class TokenType(enum.Enum):
    STRING = enum.auto()
    UINT_8 = enum.auto()
    UINT_16 = enum.auto()
    UINT_32 = enum.auto()
    PAM_SLICE_BYTES = enum.auto()
    PAM_FEATURE_BYTES = enum.auto()
    BYTES = enum.auto()

    def value_string(self, x):
        uint_types = [TokenType.UINT_8, TokenType.UINT_16, TokenType.UINT_32]
        pam_object_types = [TokenType.PAM_SLICE_BYTES,
                            TokenType.PAM_FEATURE_BYTES]

        if self is TokenType.STRING:
            return (('\'{:s}...\''.format(x[:10]))
                    if len(x) > 10 else ('\'' + x + '\''))
        elif self in uint_types:
            return '{}'.format(x)
        elif self in pam_object_types:
            return '{}'.format('[' + ', '.join(['{:d} bytes'.format(len(b))
                                                for b in x]) + ']')
        elif self is TokenType.BYTES:
            return ('[' + ', '.join([f'{b:3d}' for b in
                                     (x if len(x) <= 4 else x[:4])])
                    + (', ...' if len(x) > 4 else '')
                    + ']')


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return '{:s}({:17s}, {:})'.format(
            self.__class__.__name__,
            self.type.name,
            self.type.value_string(self.value)
        )

    @staticmethod
    def print_list(tokens):
        [print(f'{i:03d} | {t}') for i, t in enumerate(tokens)]


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

    @classmethod
    def from_binary(cls, byts):
        tokens = cls._lex_binary(byts)
        pam = cls.__init__()

        _token = tokens.pop(0)
        assert _token.type is TokenType.STRING
        pam.name = _token.value

        shape = []
        for _ in range(3):
            _token = tokens.pop(0)
            assert _token.type is TokenType.UINT_32
            shape.append(_token.value)
        pam.shape = tuple(shape)

        while tokens:
            pam.slices.append(PAMSlice.from_token(tokens.pop(0)))

        return pam

    @staticmethod
    def _lex_binary(byts):
        byte_arr = bytearray(byts)
        _next_len = pop_uint8(byte_arr)
        tokens = [Token(TokenType.STRING,
                        pop_string(byte_arr, _next_len))]
        tokens += [Token(TokenType.UINT_32,
                         pop_uint32(byte_arr)) for _ in range(3)]

        while byte_arr:
            _next_len = pop_uint32(byte_arr)
            _ = pop_word(byte_arr)
            slice_bytes = [pop_bytes(byte_arr, _next_len)]

            while True:
                current_tail = peek_bytes(byte_arr, len(PAMSlice.footer_bytes))
                if current_tail == PAMSlice.footer_bytes:
                    _ = pop_bytes(byte_arr, len(current_tail))
                    break
                else:
                    _next_len = pop_uint32(byte_arr)
                    slice_bytes.append(pop_bytes(byte_arr, _next_len))

            tokens.append(Token(TokenType.PAM_SLICE_BYTES, slice_bytes))

        return tokens


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

    @classmethod
    def from_token(cls, token):
        assert token.type is TokenType.PAM_SLICE_BYTES  # TODO - add message

        bytes_list = token.value

        if len(bytes_list) > 1:
            raise NotImplementedError
        else:
            byts = bytes_list[0]
            return cls._from_binary(byts)

    @classmethod
    def _from_binary(cls, byts):
        tokens = cls._lex_binary(byts)

    @staticmethod
    def _lex_binary(byts):
        byte_arr = bytearray(byts)
        tokens = []

        _ = pop_word(byte_arr)

        tokens.append(Token(TokenType.BYTES, pop_word(byte_arr)))

        next_len = pop_uint32(byte_arr)
        tokens.append(Token(TokenType.UINT_32, next_len))
        _ = pop_bytes(byte_arr, next_len * 2)

        while True:
            next_word = peek_word(byte_arr)
            feature_bytes = []

            if (next_word == PAMSlice.end_of_features_bytes
                    and len(byte_arr) == byte_word_length):
                if feature_bytes:
                    tokens.append(Token(TokenType.PAM_FEATURE_BYTES,
                                        feature_bytes))

                _ = pop_word(byte_arr)
                break
            elif next_word == PAMSlice.separator_bytes

        return tokens




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
    header_bytes_tail = bytes([0, 0, 16])
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

    # print(np.array([b for b in bytes(pam)]))

    filepath = os.path.join('data', 'gen_test.pam')
    pam.write_binary(filepath)

    filepath = os.path.join('data', 'test_z_mask_05.pam')
    with open(filepath, 'rb') as fle:
        byts = fle.read()
    tokens = PAM._lex_binary(byts)
    Token.print_list(tokens)


if __name__ == '__main__':
    main()
