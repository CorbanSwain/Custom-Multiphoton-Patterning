#!python3
# masks.py

import numpy as np
import imageio
import c_swain_python_utils as csutils
import functools as ft
import os


class Mask(object):
    def __init__(self, mask_data=None):
        self.mask_data = mask_data

    def create_chunk_lists(self,
                           do_points_only=False,
                           dim_priority='x',
                           do_group_chunks=True):
        print('Chunking mask file into rectangular regions ...')

        chunk_lists = []

        tot_y_size, tot_x_size, z_size = self.mask_data.shape

        for z_loc in range(z_size):
            print('> Analyzing mask at z-slice %3d of %3d'
                  % (z_loc + 1, z_size))

            zslice = self.mask_data[:, :, z_loc]
            chunk_list = ChunkList()

            if do_points_only:
                locs = np.where(np.logical_not(zslice == 0))
                for y_loc, x_loc in zip(*[l.tolist() for l in locs]):
                    new_chunk = Chunk(
                        xy_loc=(x_loc, y_loc),
                        xy_size=(1, 1),
                        total_xy_size=(tot_x_size, tot_y_size),
                        value=self.mask_data[x_loc, y_loc, z_loc])
                    chunk_list.append(new_chunk)
            else:
                if dim_priority == 'y' or dim_priority == 1:
                    dim_priority = 1
                    rowlike_iter = enumerate(zslice.T)
                elif dim_priority == 'x' or dim_priority == 0:
                    dim_priority = 0
                    rowlike_iter = enumerate(zslice)
                else:
                    raise ValueError('Unexpected dim_priority passed.')

                for a_loc, rowlike in rowlike_iter:
                    if a_loc % 100 == 99:
                        print('> > Analyzing mask at {:s} {:6,d}'.format(
                            'row' if dim_priority == 0 else 'column',
                            a_loc + 1))
                    start_idxs = np.where(np.diff(np.append(0, rowlike)))[0]
                    end_idxs = np.where(np.diff(np.append(rowlike, 0)))[0] + 1
                    widths = end_idxs - start_idxs
                    region_vals = rowlike[start_idxs]
                    filt = np.logical_not(region_vals == 0)
                    region_iter = zip(start_idxs[filt],
                                      widths[filt],
                                      region_vals[filt])

                    if do_group_chunks:
                        current_row_chunk_dict = dict()

                    for b_loc, b_size, val in region_iter:

                        if dim_priority == 1:
                            xy_loc = (a_loc, b_loc)
                            xy_size = (1, b_size)
                        else:
                            xy_loc = (b_loc, a_loc)
                            xy_size = (b_size, 1)

                        new_chunk = Chunk(
                            xy_loc=xy_loc,
                            xy_size=xy_size,
                            total_xy_size=(tot_x_size, tot_y_size),
                            value=val)

                        do_append = True

                        if do_group_chunks:
                            if a_loc > 0 and b_loc in previous_row_chunk_dict:
                                chunk = previous_row_chunk_dict[b_loc]
                                new_chk_adj = chunk.adjacency(new_chunk)

                                if dim_priority == 0 \
                                        and new_chk_adj == 'below':
                                    chunk.xy_size = (chunk.xy_size[0],
                                                     chunk.xy_size[1] + 1)
                                    new_chunk = chunk
                                    do_append = False
                                elif dim_priority == 1 \
                                        and new_chk_adj == 'right':
                                    chunk.xy_size = (chunk.xy_size[0] + 1,
                                                     chunk.xy_size[1])
                                    new_chunk = chunk
                                    do_append = False

                            current_row_chunk_dict[b_loc] = new_chunk

                        if do_append:
                            chunk_list.append(new_chunk)

                    if do_group_chunks:
                        previous_row_chunk_dict = current_row_chunk_dict


            chunk_lists.append(chunk_list)

        print('Done chunking.')
        print('Mask decomposed into {:,d} chunks.'.format(
              sum([len(cl.chunks) for cl in chunk_lists])))
        return chunk_lists

    @property
    def mask_data(self):
        return self._mask_data

    @mask_data.setter
    def mask_data(self, x):
        x_np = np.array(x)
        if x_np.ndim == 2:
            self._mask_data = x_np.reshape(x_np.shape + (1, ))
        elif x_np.ndim == 3:
            self._mask_data = x_np
        else:
            raise NotImplementedError('Mask data must be 2D or 3D')

    @classmethod
    def from_image(cls, im_path, to_float=False, to_binary=False):
        im_data = imageio.imread(im_path)

        mask_data = None

        if to_binary:
            mask_data = im_data.astype(bool)
        elif to_float:
            mask_data = im_data / np.iinfo(im_data.dtype).max

        mask_data = im_data if mask_data is None else mask_data

        return cls(mask_data)

    def write_command_files(self,
                            save_dir,
                            chunk_list_kwargs=None,
                            pam_command_kwargs=None,
                            do_optimize_file_size=True):

        print('Creating photoactivation mask file(s) for import into '
              'Prairie View.')

        chunk_list_kwargs = chunk_list_kwargs or dict()
        pam_command_kwargs = pam_command_kwargs or dict()

        if do_optimize_file_size:
            print('Optimizing file size by testing different chunking '
                  'methods.')

            from itertools import product

            if self.mask_data.dtype == 'bool':
                invert_list = [True, False]
            else:
                invert_list = [False]

            dim_priority_list = ['x', 'y']

            trial_iterator = product(invert_list, dim_priority_list)
            n_trials = len(invert_list) * len(dim_priority_list)

            best_size = None

            for i, (do_invert, dim_priority) in enumerate(trial_iterator):
                print('METHOD %d of %d: do_invert=%s, dim_priority="%s"'
                      % (i + 1,
                         n_trials,
                         'True' if do_invert else 'False',
                         dim_priority))
                if do_invert:
                    tmp_mask = Mask(np.logical_not(self.mask_data))
                else:
                    tmp_mask = self

                tmp_chunk_lists = tmp_mask.create_chunk_lists(
                    do_points_only=False,
                    dim_priority=dim_priority,
                    do_group_chunks=True)

                full_size = sum([len(cl.chunks) for cl in tmp_chunk_lists])

                if best_size is None or full_size < best_size:
                    best_size = full_size
                    did_invert = do_invert
                    chunk_lists = tmp_chunk_lists
                print('\n')

            print('BEST SIZE found to be {:,d} chunks.'.format(best_size))

        else:
            chunk_lists = self.create_chunk_lists(**chunk_list_kwargs)

        size_str = 'size={:d}x{:d}'.format(*self.mask_data.shape[:2])

        csutils.touchdir(save_dir)
        make_path = ft.partial(os.path.join, save_dir)

        print('Will save mask file(s) in directory "%s".' % save_dir)

        if did_invert:
            invert_flag = 'INVERTED_'
        else:
            invert_flag = ''

        for i, cl in enumerate(chunk_lists):
            cmd_like = cl.pam_command(**pam_command_kwargs)

            z_index = i + 1

            if type(cmd_like) is dict:
                for title, cmd_str in cmd_like.items():
                    file_name = f'{size_str}_z={z_index:03d}_{invert_flag}' \
                                f'{title}_pam.txt'
                    with open(make_path(file_name), 'w') as f:
                        f.write(cmd_str)
            else:
                cmd_str = cmd_like
                file_name = f'{size_str}_z={z_index:03d}_{invert_flag}pam.txt'
                with open(make_path(file_name), 'w') as f:
                    f.write(cmd_str)


class ChunkList(object):
    def __init__(self, chunks=None, label=None):
        self.chunks = [] if chunks is None else chunks
        self.label = label

    def __iter__(self):
        yield from self.chunks

    def __getitem__(self, item):
        return self.chunks[item]

    def __setitem__(self, key, value):
        self.chunks[key] = value

    def __add__(self, other):
        if self.label == other.label:
            new_label = self.label
        else:
            new_label = None
        return ChunkList(self.chunks + other.chunks, label=new_label)

    def append(self, item):
        self.chunks.append(item)

    def pam_command(self,
                    label_to_id=None,
                    split_values=False,
                    build_up_values=False):
        if split_values or build_up_values:
            assert split_values ^ build_up_values, \
                ('Only one of `split_values` and `build_up_values` can be set '
                 'to True.')

        kwargs = {'label': self.label, 'label_to_id': label_to_id}

        if split_values or build_up_values:
            chunk_list_map = dict()
            for c in self:
                try:
                    chunk_list_map[c.value].append(c)
                except KeyError:
                    chunk_list_map[c.value] = ChunkList()
                    chunk_list_map[c.value].append(c)

            chunk_list_command_map = {k: c.pam_command(**kwargs)
                                      for k, c in chunk_list_map.items()}

            output_dict = dict()
            if split_values:
                for val, command in chunk_list_command_map:
                    output_dict[f'value={val:d}'] = command

            elif build_up_values:
                values = reversed(sorted(chunk_list_map.keys()))
                deltas = np.diff([0] + values)
                for i, d in enumerate(deltas):
                    commands = [chunk_list_command_map[v]
                                for v in values[i:]]
                    command = ', '.join(commands)
                    output_dict[f'layer={i:03d}_delta={d:+03d}'] = command

            return output_dict

        else:
            return ', '.join(c.pam_command(**kwargs) for c in self)


class Chunk(object):
    def __init__(self, xy_loc, xy_size, total_xy_size, value=1, label=None):
        self.xy_loc = xy_loc
        self.xy_size = xy_size
        self.total_xy_size = total_xy_size
        self.value = value
        self.label = label

        self.inset_subpixel_width = 1e-1

    @property
    def _n_decimals(self):
        return int(np.ceil(np.log10(np.max(
            self.total_xy_size, keepdims=False) / self.inset_subpixel_width)))

    @property
    def hull_points_relative(self):
        total_size = np.array(self.total_xy_size)
        isw = self.inset_subpixel_width

        top_left = (np.array(self.xy_loc) + isw) / total_size
        size = ((np.array(self.xy_size) - (2 * isw))
                / total_size)
        bottom_right = top_left + size

        top_left = np.round(top_left, self._n_decimals)
        bottom_right = np.round(bottom_right, self._n_decimals)

        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        return [tuple(top_left.tolist()),
                top_right,
                tuple(bottom_right.tolist()),
                bottom_left]

    @property
    def hull_points(self):
        xy_size = self.xy_size
        top_left = self.xy_loc
        bottom_right = (top_left[0] + xy_size[0],
                        top_left[1] + xy_size[1])
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        return [top_left,
                top_right,
                bottom_right,
                bottom_left]

    @property
    def top_left(self):
        return self.xy_loc

    @property
    def top_right(self):
        return (self.xy_loc[0] + self.xy_size[0],
                self.xy_loc[1])

    @property
    def bottom_right(self):
        return (self.xy_loc[0] + self.xy_size[0],
                self.xy_loc[1] + self.xy_size[1])

    @property
    def bottom_left(self):
        return (self.xy_loc[0],
                self.xy_loc[1] + self.xy_size[1])

    def pam_command(self, label=None, label_to_id=None):
        label = self.label if label is None else label

        id_int = 1 if label is None else None
        if id_int is None:
            id_int = label_to_id[label]

        id_int_str = f'{(-1 * id_int):d}'

        return ', '.join(['%.*f' % (self._n_decimals, x)
                          for point in self.hull_points_relative
                          for x in point]
                         + [id_int_str])

    def adjacency(self, other, epsilon=1e-9):
        adj_result, do_reverse_eval = self._adjacency_helper(other,
                                                             epsilon=epsilon)

        if do_reverse_eval:
            assert adj_result is None, 'Unexpected condition met.'

            adj_result, _ = other._adjacency_helper(self,
                                                    epsilon=epsilon)
            if adj_result == 'above':
                return 'below'
            if adj_result == 'right':
                return 'left'
            else:
                return adj_result
        else:
            return adj_result

    def _adjacency_helper(self, other, epsilon):
        def _eq(a, b):
            # a = np.array(a)
            # b = np.array(b)
            # return np.all(np.abs(a - b) < epsilon)
            return a == b

        def _peq(a, b):
            return a[0] == b[0] and a[1] == b[1]

        if _eq(self.value, other.value):
            above_check = (_peq(self.top_left, other.bottom_left)
                           and _peq(self.top_right, other.bottom_right))
            if above_check:
                return 'above', False

            right_check = (_peq(self.top_right, other.top_left)
                           and _peq(self.bottom_right, other.bottom_left))
            if right_check:
                return 'right', False

            return None, True
        else:
            return None, False

    def __repr__(self):
        return '''{:s}(
    xy_loc=({:s}), 
    xy_size=({:s}), 
    total_xy_size=({:s}), 
    value={:f}, 
    label={:s})'''.format(
            self.__class__.__name__,
            ', '.join('{:d}'.format(x) for x in self.xy_loc),
            ', '.join('{:d}'.format(x) for x in self.xy_size),
            ', '.join('{:d}'.format(x) for x in self.total_xy_size),
            self.value,
            'None' if self.label is None else self.label)


def main():
    filename = 'm77.png'
    im_path = os.path.join('data', filename)
    mask = Mask.from_image(im_path, to_binary=True)
    mask.write_command_files(
        os.path.join('data', 'generated_masks', filename + '_commands'),
        do_optimize_file_size=True)


if __name__ == '__main__':
    main()
