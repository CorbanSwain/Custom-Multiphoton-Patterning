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

    def create_chunk_lists(self, do_points_only=False):
        chunk_lists = []

        tot_y_size, tot_x_size, z_size = self.mask_data.shape

        for z_loc in range(z_size):
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
                for y_loc, row in enumerate(zslice):
                    start_idxs = np.where(np.diff(np.append(0, row)))[0]
                    end_idxs = np.where(np.diff(np.append(row, 0)))[0] + 1
                    widths = end_idxs - start_idxs
                    region_vals = row[start_idxs]
                    filt = np.logical_not(region_vals == 0)
                    iterator = zip(start_idxs[filt],
                                   widths[filt],
                                   region_vals[filt])

                    for x_loc, x_size, val in iterator:
                        new_chunk = Chunk(
                            xy_loc=(x_loc, y_loc),
                            xy_size=(x_size, 1),
                            total_xy_size=(tot_x_size, tot_y_size),
                            value=val)
                        chunk_list.append(new_chunk)

            chunk_lists.append(chunk_list)

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
                            pam_command_kwargs=None):

        chunk_list_kwargs = chunk_list_kwargs or dict()
        pam_command_kwargs = pam_command_kwargs or dict()

        chunk_lists = self.create_chunk_lists(**chunk_list_kwargs)
        size_str = 'size={:d}x{:d}'.format(*self.mask_data.shape[:2])

        csutils.touchdir(save_dir)
        make_path = ft.partial(os.path.join, save_dir)

        for i, cl in enumerate(chunk_lists):
            cmd_like = cl.pam_command(**pam_command_kwargs)
            z_index = i + 1

            if type(cmd_like) is dict:
                for title, cmd_str in cmd_like.items():
                    file_name = f'{size_str}_z={z_index:03d}_{title}_pam.txt'
                    with open(make_path(file_name), 'w') as f:
                        f.write(cmd_str)
            else:
                cmd_str = cmd_like
                file_name = f'{size_str}_z={z_index:03d}_pam.txt'
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

    @property
    def hull_points_relative(self):
        total_size = np.array(self.total_xy_size)
        inset_subpixel_width = 5e-3
        top_left = (np.array(self.xy_loc) + inset_subpixel_width) / total_size
        size = ((np.array(self.xy_size) - (2 * inset_subpixel_width))
                / total_size)
        bottom_right = top_left + size
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        return [tuple(top_left.tolist()),
                top_right,
                tuple(bottom_right.tolist()),
                bottom_left]

    def pam_command(self, label=None, label_to_id=None):
        label = self.label if label is None else label

        id_int = 1 if label is None else None
        if id_int is None:
            id_int = label_to_id[label]

        id_int_str = f'{(-1 * id_int):d}'

        return ', '.join([f'{x:.6f}'
                          for point in self.hull_points_relative
                          for x in point]
                         + [id_int_str])


def main():
    im_path = os.path.join('data', 'test_mask_55.png')
    mask = Mask.from_image(im_path)
    mask.write_command_files(
        os.path.join('data', 'generated_masks', 'mask_55_gen_07'))


if __name__ == '__main__':
    main()
