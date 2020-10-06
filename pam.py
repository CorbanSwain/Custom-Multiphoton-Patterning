#!python3
# pam.py

import numpy as np
import os
import enum
import functools as ft
import struct



default_encoding = 'utf8'
default_byteorder = 'little'  # ... endian
byte_word_length = 4


def isint(x):
    return round(x) == x


def iseven(x):
    return (x % 2) == 0


indent_str = ' ' * 4


def indent(string):
    return indent_str + (('\n' + indent_str).join(string.splitlines()))


def bytes_to_binary_str(byts):
    return ' '.join(format(b, '#010b')[2:] for b in byts)


def bytes_to_int_arr(byts):
    return [b for b in byts]


def bytes_xor(a, b):
    return bytes([_a ^ _b for _a, _b in zip(a, b)])


def bytes_sum(a, b):
    output_len = max(len(a), len(b))
    return uint_to_bytes(bytes_to_uint(a) + bytes_to_uint(b),
                         length=output_len,
                         allow_overflow=True)


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
            self.type.value_string(self.value))


def print_token_list(tokens):
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
        return '''{:s}(name='{:s}',
    shape={}, 
    slices=[{:s}])'''.format(
            self.__class__.__name__,
            self.name,
            self.shape,
            ('' if not self.slices else
             ('\n' + indent(',\n'.join(indent(f'{s}') for s in self.slices)))))

    def write_binary(self, filepath):
        with open(filepath, 'wb') as fle:
            fle.write(bytes(self))

    def check_consistency(self):
        pass  # TODO

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'rb') as fle:
            byts = fle.read()
        return cls.from_binary(byts)

    @classmethod
    def from_binary(cls, byts):
        tokens = cls._lex_binary(byts)
        pam = cls()

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

    def __init__(self, features=None):
        self.features = [] if features is None else features

        self._unknown_bytes_raw = None
        self._tail_bytes_raw = None

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
        return '''{:s}(
    features=[{:s}])'''.format(
            self.__class__.__name__,
            ('' if not self.features else
             ('\n' + indent(',\n'.join(indent(f'{s}')
                                       for s in self.features)))))

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

        pam_slice = cls()

        _next_token = tokens.pop(0)
        assert _next_token.type is TokenType.BYTES
        pam_slice._unknown_bytes_raw = _next_token.value
        pam_slice._tail_bytes_raw = byts[4:]

        _next_token = tokens.pop(0)
        assert _next_token.type is TokenType.UINT_32
        assert iseven(_next_token.value)
        num_features = _next_token.value // 2 + 1

        assert len(tokens) == num_features

        while tokens:
            pam_slice.features.append(PAMFeature.from_token(tokens.pop(0)))

        return pam_slice

    @staticmethod
    def _lex_binary(byts):
        byte_arr = bytearray(byts)
        tokens = [Token(TokenType.BYTES, pop_word(byte_arr))]

        _ = pop_word(byte_arr)

        next_len = pop_uint32(byte_arr)
        tokens.append(Token(TokenType.UINT_32, next_len))
        _ = pop_bytes(byte_arr, next_len * 2)

        feature_bytes_list = []
        feature_bytes = pop_word(byte_arr)
        while True:
            next_word = peek_word(byte_arr)

            if (next_word == PAMSlice.end_of_features_bytes
                    and len(byte_arr) == byte_word_length):
                if feature_bytes:
                    feature_bytes_list.append(feature_bytes)
                    tokens.append(Token(TokenType.PAM_FEATURE_BYTES,
                                        feature_bytes_list))

                _ = pop_word(byte_arr)
                break

            elif next_word.endswith(PAMFeature.header_tail_bytes):
                feature_bytes_list.append(feature_bytes)
                tokens.append(Token(TokenType.PAM_FEATURE_BYTES,
                                    feature_bytes_list))
                feature_bytes_list = []
                feature_bytes = pop_word(byte_arr)

            else:
                feature_bytes += pop_word(byte_arr)

        return tokens

    @property
    def num_features(self):
        return len(self.features)

    @property
    def _unknown_bytes(self):
        unknown_bytes = None

        if self._unknown_bytes_raw:
            unknown_bytes = self._unknown_bytes_raw
        elif self._isempty:
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
        n = (self.num_features - 1)
        assert n >= 0, 'features list must have at least one item.'
        return uint32_to_bytes(n * 2) + (uint32_to_bytes(2) * n)


class PAMFeature(object):
    header_tail_bytes = bytes([0, 0, 16])
    header_bytes = None
    tail_bytes = None

    def __init__(self, header_bytes=None, tail_bytes=None):
        if header_bytes is not None:
            self.header_bytes = bytes(header_bytes)

        if tail_bytes is not None:
            self.tail_bytes = bytes(tail_bytes)

    def __bytes__(self):
        return self.header_bytes + self.tail_bytes

    def __repr__(self):
        return '''{:s}(header_bytes={}, tail_bytes={:s})'''.format(
            self.__class__.__name__,
            (None if self.header_bytes is None else
             [b for b in self.header_bytes]),
            (repr(None) if self.tail_bytes is None else
             ('[' + ', '.join(f'{b}' for b in self.tail_bytes[:4]) + ', ...]'
              if len(self.tail_bytes) > 4
              else f'{[b for b in self.tail_bytes]}')))

    @classmethod
    def from_token(cls, token):
        assert token.type is TokenType.PAM_FEATURE_BYTES

        if not len(token.value) == 1:
            raise NotImplementedError

        byts = token.value[0]
        tokens = cls._lex_binary(byts)

        _next_token = tokens.pop(0)
        assert _next_token.type is TokenType.BYTES
        header_bytes = _next_token.value

        _next_token = tokens.pop(0)
        assert _next_token.type is TokenType.BYTES
        tail_bytes = _next_token.value

        feature_class = cls.header_bytes_to_class(header_bytes)
        return feature_class.from_binary(header_bytes, tail_bytes)

    @classmethod
    def from_binary(cls, header_bytes, tail_bytes):
        return cls(header_bytes, tail_bytes)

    @staticmethod
    def _lex_binary(byts):
        byte_arr = bytearray(byts)
        tokens = [Token(TokenType.BYTES, pop_word(byte_arr)),
                  Token(TokenType.BYTES, pop_bytes(byte_arr, len(byte_arr)))]
        return tokens

    @staticmethod
    def header_bytes_to_class(header_bytes):
        if header_bytes == RectanglePAMFeature.header_bytes:
            return RectanglePAMFeature
        elif header_bytes == EmptyPAMFeature.header_bytes:
            return EmptyPAMFeature
        elif header_bytes == InvertPAMFeature.header_bytes:
            return InvertPAMFeature
        else:
            return PAMFeature

    @staticmethod
    def point_to_bytes(point):
        byte_list = [uint16_to_bytes(x) for x in point]
        return bytes().join(byte_list)


class RectanglePAMFeature(PAMFeature):
    header_bytes = bytes([1, 0, 0, 16])
    separator_bytes = bytes([0, 64, 0, 0])
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

    @classmethod
    def from_binary(cls, _, tail_bytes):
        tokens = cls._lex_binary(tail_bytes)

        # FIXME - some of this should be moved to a PolygonPAMFeature class
        _next_token = tokens.pop(0)
        assert _next_token.type == TokenType.UINT_32, \
            f'Error while parsing binary. Expected next token to be UINT_32.'
        num_points = _next_token.value

        if num_points == 4:
            point_list = []
            for _ in range(num_points):
                _next_token = tokens.pop(0)
                assert _next_token.type == TokenType.UINT_16
                x_val = _next_token.value

                _next_token = tokens.pop(0)
                assert _next_token.type == TokenType.UINT_16
                y_val = _next_token.value

                point_list.append(np.array([x_val, y_val]))

            # TODO - add validity checks for rectangle

            top_left = point_list[0]
            size = point_list[2] - top_left

            if (size == [1, 1]).all():
                return PointPAMFeature(point=top_left)
            else:
                return cls(top_left_point=top_left, size=size)
        else:
            print(f'Parsing failed. Expected 4 points, but got {num_points} '
                  f'points instead. Returning general PAMFeature instance.')
            return PAMFeature(cls.header_bytes, tail_bytes)

    @classmethod
    def _lex_binary(cls, byts):
        byte_arr = bytearray(byts)
        tokens = []

        _next_len = pop_uint32(byte_arr)
        err_header = 'Error while lexing binary to generate tokens for a ' \
                     'PAMFeature.'
        assert len(byte_arr) == _next_len, err_header + \
            f'Expected len of {_next_len:d} bytes; found len of '\
            f'{len(byte_arr):d} bytes instead.'

        _next_word = pop_word(byte_arr)
        assert _next_word == PAMSlice.separator_bytes  # TODO - add message

        _next_len = pop_uint32(byte_arr)
        tokens.append(Token(TokenType.UINT_32, _next_len))

        _next_word = pop_word(byte_arr)
        assert _next_word == cls.separator_bytes, err_header + \
            f'Expected next bytes to be {[b for b in cls.separator_bytes]}; ' \
            f'instead got {[b for b in _next_word]}.'

        for _ in range(_next_len):
            tokens += [Token(TokenType.UINT_16, pop_uint16(byte_arr))
                       for _ in range(2)]

        _next_word = peek_word(byte_arr)
        if _next_word == cls.footer_bytes:
            _ = pop_word(byte_arr)
        elif len(_next_word) == 0:
            pass
        else:
            # FIXME - use function for byte arr conversion
            raise AssertionError(f'Expected end of binary, however bytes '
                                 f'remain: {[b for b in byte_arr]}.')

        return tokens

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
        return self.top_left_point + (self.size[0] * np.array([1, 0]))

    @property
    def bottom_right_point(self):
        return self.top_left_point + self.size

    @property
    def bottom_left_point(self):
        return self.top_left_point + (self.size[1] * np.array([0, 1]))

    @property
    def _all_points(self):
        return [self.top_left_point,
                self.top_right_point,
                self.bottom_right_point,
                self.bottom_left_point]

    @property
    def num_points(self):
        return len(self._all_points)

    @property
    def tail_bytes(self):
        tail_bytes_list = (
                [PAMSlice.separator_bytes,
                 uint32_to_bytes(self.num_points),
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

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @classmethod
    def from_binary(cls, _, tail_bytes):
        assert len(tail_bytes) == 0
        return cls()


class InvertPAMFeature(PAMFeature):
    header_bytes = bytes([0, 0, 0, 16])
    tail_bytes = bytes(([0] * 8) + ([0, 0, 0, 67] * 2))

    def __repr__(self):
        return self.__class__.__name__ + '()'

    @classmethod
    def from_binary(cls, _, tail_bytes):
        assert tail_bytes == cls.tail_bytes
        return cls()


def gen_test():
    pam = PAM(name='gentest_z_mask_08',
              shape=(128, 128, 5),
              slices=[
                  PAMSlice(
                      features=[
                          EmptyPAMFeature()]),
                  PAMSlice(
                      features=[
                          EmptyPAMFeature()]),
                  PAMSlice(
                      features=[
                          PointPAMFeature(point=[0, 0])]),
                  PAMSlice(
                      features=[
                          RectanglePAMFeature(top_left_point=[0, 0],
                                              size=[2, 1])]),
                  PAMSlice(
                      features=[
                          RectanglePAMFeature(top_left_point=[1, 0],
                                              size=[5, 2]),
                          RectanglePAMFeature(top_left_point=[10, 1],
                                              size=[6, 1])])])
    filepath = os.path.join('data', 'gentest_z_mask_08.pam')
    pam.write_binary(filepath)

    pam = PAM(name='gentest_mask_11',
              shape=(256, 256, 1))
    x = np.arange(32, 64)
    y = np.arange(128, 225)
    features = []
    for x_val in x:
        for y_val in y:
            features.append(PointPAMFeature(point=[x_val, y_val]))

    pam.slices.append(PAMSlice(features=features))

    filepath = os.path.join('data', 'gentest_mask_11.pam')
    pam.write_binary(filepath)

    print(pam)


def main():
    pam = PAM(name='cool_pam', shape=(64, 64, 2))
    pam.slices = [PAMSlice() for _ in range(pam.shape[2])]
    pam.slices[0].features.append(EmptyPAMFeature())
    pam.slices[1].features.append(PointPAMFeature(point=(1, 1)))

    # print(np.array([b for b in bytes(pam)]))

    filepath = os.path.join('data', 'gen_test.pam')
    pam.write_binary(filepath)

    filepath = os.path.join('data', 'test_z_mask_08.pam')
    with open(filepath, 'rb') as fle:
        byts = fle.read()

    pam_2 = PAM.from_file(filepath)
    print(pam_2)

    read_bytes = np.array([b for b in byts])
    print('\nRead Bytes:')
    print(read_bytes)

    gen_bytes = np.array([b for b in bytes(pam_2)])
    print('\nGenerated Bytes:')
    print(gen_bytes)

    print(f'\nNumber of differences = '
          f'{np.sum(np.logical_not(read_bytes == gen_bytes))}')

    results_dict = {
        'x': [],
        'y': [],
        'xor': [],
        'xor_2': [],
        'sum': [],
        'sum_2': []
    }

    for slce in pam_2.slices:
        x = slce._unknown_bytes_raw
        y = slce._tail_bytes_raw
        y_copy = bytearray(y)
        y_chunk = []
        while y_copy:
            y_chunk.append(pop_word(y_copy))

        y_xor = ft.reduce(bytes_xor, y_chunk)
        y_xor_2 = bytes_xor(y_xor, x)
        y_sum = ft.reduce(bytes_sum, y_chunk)
        y_sum_2 = bytes_sum(y_sum, x)

        results_dict['x'].append(x)
        results_dict['y'].append(y)
        results_dict['xor'].append(y_xor)
        results_dict['xor_2'].append(y_xor_2)
        results_dict['sum'].append(y_sum)
        results_dict['sum_2'].append(y_sum_2)

    print(f'\nunknown bytes')
    [print(bytes_to_binary_str(x)) for x in results_dict['x']]

    print(f'\nunknown bytes (as floats)')
    [print(struct.unpack('f', x)[0]) for x in results_dict['x']]

    # print(f'\nXOR reduction')
    # [print(bytes_to_binary_str(x)) for x in results_dict['xor']]
    #
    # print(f'\nXOR reduction XOR\'d with unknown')
    # [print(bytes_to_binary_str(x)) for x in results_dict['xor_2']]

    print(f'\nSUM reduction')
    [print(bytes_to_binary_str(x)) for x in results_dict['sum']]

    # print(f'\nSUM reduction SUM\'d with unknown')
    # [print(bytes_to_binary_str(x)) for x in results_dict['sum_2']]

    gen_test()




if __name__ == '__main__':
    main()
