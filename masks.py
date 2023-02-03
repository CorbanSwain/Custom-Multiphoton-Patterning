#!python3
# masks.py
import shutil

import numpy as np
import imageio
import c_swain_python_utils as csutils
import functools as ft
import os
import re
from math import ceil
import matplotlib.pyplot as plt
import pprint as pp


class Mask(object):
    save_dir_tag = 'mask_outputs'

    def __init__(self, mask_data=None, load_path=None):
        self.mask_data = mask_data
        self.load_path = load_path

        self._save_dir = None

    def create_chunk_lists(self,
                           do_points_only=False,
                           dim_priority='x',
                           do_group_chunks=True):
        print('Chunking mask file into rectangular regions ...')

        chunk_lists = []

        z_size, tot_y_size, tot_x_size = self.mask_data.shape

        for z_loc, zslice in enumerate(self.mask_data):
            print('> Analyzing mask at z-slice %3d of %3d'
                  % (z_loc + 1, z_size))

            chunk_list = ChunkList()

            if do_points_only:
                locs = np.where(np.logical_not(zslice == 0))
                for y_loc, x_loc in zip(*[l.tolist() for l in locs]):
                    new_chunk = Chunk(
                        xy_loc=(x_loc, y_loc),
                        xy_size=(1, 1),
                        total_xy_size=(tot_x_size, tot_y_size),
                        value=self.mask_data[z_loc, x_loc, y_loc])
                    chunk_list.append(new_chunk)
            else:
                if dim_priority == 'y' or dim_priority == 1:
                    dim_priority = 1
                    rowlike_iter = enumerate(zslice.T)
                elif dim_priority == 'x' or dim_priority == 0:
                    dim_priority = 0
                    rowlike_iter = enumerate(zslice)
                else:
                    raise ValueError('Unexpected `dim_priority` passed.')

                for a_loc, rowlike in rowlike_iter:
                    if a_loc % 100 == 99:
                        print('> > Analyzing mask at {:s} {:6,d}'.format(
                            'row' if dim_priority == 0 else 'column',
                            a_loc + 1))
                    ref_idxs = np.where(np.diff(rowlike))[0] + 1
                    start_idxs = np.append(0, ref_idxs)
                    end_idxs = np.append(ref_idxs, rowlike.shape[0])
                    widths = end_idxs - start_idxs
                    region_vals = rowlike[start_idxs]
                    filt = np.logical_not(region_vals == 0)
                    region_iter = zip(start_idxs[filt],
                                      widths[filt],
                                      region_vals[filt])

                    if do_group_chunks:
                        current_row_chunk_dict = dict()

                    for b_loc, b_size, val in region_iter:

                        if dim_priority == 0:
                            xy_loc = (b_loc, a_loc)
                            xy_size = (b_size, 1)
                        else:
                            xy_loc = (a_loc, b_loc)
                            xy_size = (1, b_size)

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
            self._mask_data = x_np.reshape((1, ) + x_np.shape)
        elif x_np.ndim == 3:
            self._mask_data = x_np
        else:
            raise NotImplementedError('Mask data must be 2D or 3D.')

    @property
    def save_dir(self):
        if self._save_dir is None:
            if self.load_path is None:
                return os.path.join('data', self.save_dir_tag)
            else:
                return '_'.join([self.load_path, self.save_dir_tag])
        else:
            return self._save_dir

    @save_dir.setter
    def save_dir(self, x):
        self._save_dir = x

    @classmethod
    def from_image(cls, im_path, to_float=False, to_binary=False):
        try:
            im_data = imageio.volread(im_path)
        except ValueError:
            im_data = imageio.imread(im_path)

        mask_data = None

        if to_binary:
            mask_data = im_data.astype(bool)
        elif to_float:
            mask_data = im_data / np.iinfo(im_data.dtype).max

        mask_data = im_data if mask_data is None else mask_data

        print('Loaded image has shape: \n')
        pp.pprint(mask_data.shape)

        return cls(mask_data=mask_data, load_path=im_path)

    @csutils.timed()
    def write_command_files(self,
                            save_dir=None,
                            chunk_list_kwargs=None,
                            pam_command_kwargs=None,
                            do_optimize_file_size=True,
                            do_split_files=False,
                            file_split_size_mb=0.25,
                            do_preview=True,
                            do_allow_binary_inversion=False,
                            auto_set_label_to_id=True,
                            max_num_output_values=8,
                            do_prepare_sequential_import=True):

        if do_split_files:
            file_split_size = file_split_size_mb * 1e6

        print('Creating photoactivation mask file(s) for import into '
              'Prairie View.')

        chunk_list_kwargs = chunk_list_kwargs or dict()
        pam_command_kwargs = pam_command_kwargs or dict()

        if do_optimize_file_size:
            print('\nOptimizing file size by testing different chunking '
                  'methods.')

            from itertools import product

            if self.mask_data.dtype == 'bool':
                if do_allow_binary_inversion:
                    invert_list = [False, True]
                else:
                    invert_list = [False]
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
                         do_invert,
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
                    best_method = i
                    best_size = full_size
                    did_invert = do_invert
                    chunk_lists = tmp_chunk_lists
                print('\n')

            print('BEST SIZE found to be {:,d} chunks with method # {:d}.'
                  .format(best_size, best_method + 1))

        else:
            chunk_lists = self.create_chunk_lists(**chunk_list_kwargs)

        size_str = 'size={1:d}x{0:d}'.format(*self.mask_data.shape[1:])

        if save_dir is None:
            save_dir = self.save_dir

        csutils.touchdir(save_dir)
        make_path = ft.partial(os.path.join, save_dir)

        print('Will save mask file(s) in directory "%s".\n' % save_dir)

        if do_optimize_file_size and did_invert:
            invert_flag = 'INVERTED_'
        else:
            invert_flag = ''

        if auto_set_label_to_id:
            chunk_value_set = set()
            for cl in chunk_lists:
                for c in cl:
                    chunk_value_set.add(c.value)

            chunk_value_list = sorted(chunk_value_set)
            if len(chunk_value_list) > max_num_output_values:
                print(f'Digitizing `Chunk` values to have only '
                      f'{max_num_output_values:d} levels')
                values_np = np.array(chunk_value_list)
                bins = np.linspace(values_np[0], values_np[-1],
                                   max_num_output_values)
                values_digitized = np.digitize(values_np, bins)
                id_to_value_dict = dict()
                for i in range(max_num_output_values):
                    id_int = i + 1
                    id_to_value_dict[id_int] = np.median(
                        values_np[values_digitized == id_int])

                values_digitized = values_digitized.tolist()
                label_to_id = {v: _id for v, _id in zip(chunk_value_list,
                                                        values_digitized)}

            else:
                label_to_id = {v: (i + 1)
                               for i, v in enumerate(chunk_value_list)}
                id_to_value_dict = {v: k for k, v in label_to_id.items()}

            if 'label_to_id' in pam_command_kwargs:
                print('WARNING: overwriting `label_to_id` parameter in '
                      '`pam_command_kwargs`.')

            pam_command_kwargs['label_to_id'] = label_to_id

            print('Auto-set `label_to_id` dict based on values of entire mask.')

            print('Writing file with conversion factors between palette '
                  'indices and laser powers.')
            firstlines = [f'Palette Conversion Table (generated at '
                          f'{csutils.nowstr()})\n',
                          '| Palette Index | Relative Laser Power | '
                          'Mask Value |\n',
                          f'| {"---":>13s} | {"---":>20s} | {"---":>10s} |\n']

            image_vals = np.array(list(id_to_value_dict.values()))
            minval = min(image_vals[image_vals > 0])

            lines = [f'| {k:13d} | {v / minval:20.3f} | {v:10.3f} |\n'
                     for k, v in id_to_value_dict.items()]

            file_name = 'palette-power_conversion_table.txt'
            file_path = make_path(file_name)

            with open(file_path, 'w') as f:
                f.writelines(firstlines + lines)

            print(f'Wrote lookup table to "{file_path:s}":')
            print(''.join(firstlines + lines))
            print('\n')
        else:
            # TODO - Handle max output values when not auto-setting label to id.
            #        Maybe need to pass param to the `cl.pam_command()`
            #        function.
            print('WARNING: **Not Implemented**, `max_num_output_values` '
                  'parameter will be ignored since `auto_set_label_to_id` is '
                  'False.')

        file_paths = []

        for i, cl in enumerate(chunk_lists):
            cmd_like = cl.pam_command(**pam_command_kwargs)

            z_index = i + 1

            if type(cmd_like) is dict:
                if do_prepare_sequential_import:
                    print('NOT IMPLEMENTED: Cannot prepare sequential import.')
                    do_prepare_sequential_import = False

                for title, cmd_str in cmd_like.items():
                    file_name = f'{size_str}_z={z_index:03d}_{invert_flag}' \
                                f'{title}_pam.txt'
                    file_path = make_path(file_name)
                    print(file_path)
                    with open(file_path, 'w') as f:
                        f.write(cmd_str)

                    if do_preview:
                        preview_command_file(
                            file_path,
                            self.mask_data.shape[1:])

                    if do_split_files:
                        split_file(file_path,
                                   file_split_size,
                                   preview_shape=(self.mask_data.shape[1:]
                                                  if do_preview else None))
            else:
                cmd_str = cmd_like
                file_name = f'{size_str}_z={z_index:03d}_{invert_flag}pam.txt'
                file_path = make_path(file_name)
                print(file_path)
                with open(file_path, 'w') as f:
                    f.write(cmd_str)

                if do_preview:
                    preview_command_file(
                        file_path,
                        self.mask_data.shape[1:])

                if do_split_files:
                    split_file(file_path,
                               file_split_size,
                               preview_shape=(self.mask_data.shape[1:]
                                              if do_preview else None))

            file_paths.append((z_index, file_path))

        if do_prepare_sequential_import:
            print('\nCopying PAM files to subfolder to prepare for sequential '
                  'z-stack import.')

            seq_dir = make_path(f'SEQ_IMPORT_{size_str}x{len(file_paths)}_'
                                f'{invert_flag}pams')
            print(f'Sequential import directory: "{seq_dir:s}"')
            if os.path.exists(seq_dir):
                shutil.rmtree(seq_dir)
            csutils.touchdir(seq_dir)
            for z_index, file_path in file_paths:
                new_path = os.path.join(seq_dir, f'{z_index:03d}.txt')
                shutil.copyfile(file_path, new_path)

        print('\nMask command file(s) generation complete!')


def shift_pixel_edges(x,
                      axis=1,
                      *,
                      leading_edge_shift,
                      falling_edge_shift,
                      warn=True,
                      verbose=False):
    assert x.ndim == 2, 'Array must be exactly 2-dimensional'

    transpose = axis == 0
    if transpose:
        x = x.T

    # - Get All edges with diff
    # - Arrange edges into big vector from all image rows
    #

    diff_arr = np.diff(x, axis=1)
    le_index_tup = np.where(diff_arr > 0)  # leading edges
    fe_index_tup = np.where(diff_arr < 0)  # falling edges
    le_ri, le_ci = le_index_tup
    fe_ri, fe_ci = fe_index_tup

    for row in range(x.shape[0]):
        le = le_ci[le_ri == row]
        fe = le_ci[fe_ri == row]
        all_e_raw = np.concatenate([le, fe])
        if all_e_raw.size == 0:
            continue
        e_argsort = np.argsort(all_e_raw)
        all_e = all_e_raw[e_argsort]

    le_mask_raw = np.concatenate([np.ones(le.shape, dtype=bool),
                                  np.ones(fe.shape, dtype=bool)])
    le_mask = le_mask_raw[e_argsort]
    fe_mask = np.logical_not(le_mask)

    new_le = le + leading_edge_shift
    new_fe = fe + falling_edge_shift

    x_prime = np.array()

    if transpose:
        x_prime = x_prime.T
    return x_prime




def split_file(pth, max_size, preview_shape=None):
    full_file_size = os.path.getsize(pth)
    num_outfiles = ceil(full_file_size / max_size)

    max_size_mb = max_size / 1e6

    if num_outfiles > 1:

        print('Splitting output file into %d files of approx %.2f MB.\n'
              % (num_outfiles, max_size_mb))

        with open(pth, 'r') as ref_file:
            ref_file_text = ref_file.read()

        tokens = ref_file_text.split(',')
        ref_file_lines = []
        temp_line = ''
        for t in tokens:
            t = t.strip()
            if temp_line == '':
                temp_line = t
                continue

            temp_line = ','.join([temp_line, t])
            if float(t) < 0:
                ref_file_lines.append(temp_line + ',')
                temp_line = ''

        n_ref_file_lines = len(ref_file_lines)
        lines_per_file = n_ref_file_lines // num_outfiles
        tmp_total_lines = lines_per_file * num_outfiles
        n_last_file_lines = (lines_per_file
                             + (n_ref_file_lines - tmp_total_lines))
        split_lens = (([lines_per_file,] * (num_outfiles - 1))
                      + [n_last_file_lines,])

        split_dir = pth + (' SPLIT (at %.2f MB)' % max_size_mb)
        csutils.touchdir(split_dir)

        _, _tail = os.path.split(pth)
        path_fmt = split_dir + os.sep + '{:03d}.txt'

        start_idx = 0
        for i_file, file_len in enumerate(split_lens):
            end_idx = start_idx + file_len
            file_lines = ref_file_lines[start_idx:end_idx]
            last_line = file_lines[-1]
            if last_line.endswith(','):
                last_line = last_line[:-1]
            file_lines[-1] = last_line
            with open(path_fmt.format(i_file + 1), 'w') as f:
                f.write(''.join(file_lines))
            start_idx = end_idx

            if preview_shape is not None:
                preview_command_file(
                    path_fmt.format(i_file + 1),
                    preview_shape)

        assert start_idx == n_ref_file_lines, 'Unexpected error when splitting.'

    else:
        print('No file splitting performed since the input file is under the '
              '%.2f MB limit.' % max_size_mb)


def preview_command_file(file_path,
                         image_size=(1, 1),
                         verbose=False):
    try:
        print('\nAttempting to preview command file "%s"' % file_path)

        with open(file_path, 'r') as f:
            file_lines = f.readlines()

        figure_dir, filename = os.path.split(file_path)
        figure_savename = '%s_preview' % filename
        figure_name = 'PAM Command "%s" preview' % filename

        file_text = ''.join(file_lines)
        clean_text = re.sub(r'\s+', '', file_text, flags=re.UNICODE)
        tokens = clean_text.split(',')

        # csutils.set_mpl_defaults()
        f = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.set_aspect('equal')

        if len(tokens) <= 2:
            print(f'Insufficient number of values ({len(tokens):d}) found in '
                  f'command file to produce any mask shapes.')
            plt.text(0.5, 0.5, '( Empty Mask File )',
                     transform=ax.transAxes,
                     ha='center',
                     va='center',
                     c='black',
                     alpha=0.9,
                     fontsize='x-large',
                     fontweight='bold')
        else:
            image_size = np.array(image_size)
            xy_size = np.flip(image_size)

            chunks = []
            current_chunk = []
            for t in tokens:
                float_t = float(t)
                if float_t < 0:
                    points = np.array(current_chunk)
                    points = points.reshape((-1, 2)) * xy_size
                    chunks.append((points, -1 * int(t)))
                    current_chunk = []
                else:
                    current_chunk.append(float_t)

            print_prd = ceil(len(chunks) / 3)
            for i, c in enumerate(chunks):
                if verbose or (i % print_prd) == (print_prd - 1):
                    print('Chunk {:6,d} : {:s}'.format(i + 1, str(c)))

            print('Creating Figure')
            legend_handles = dict()
            for i, c in enumerate(chunks):
                point_list, label_int = c
                x, y = (point_list[:, 0], point_list[:, 1])
                plt_handle, = ax.fill(x, y,
                                     facecolor='C{:d}'.format(label_int - 1),
                                     linestyle='-',
                                     linewidth=0.035,
                                     edgecolor='k')

                legend_handles.setdefault(label_int, (plt_handle,
                                                      f'Palette Index '
                                                      f'{label_int:d}'))

            ax.legend(*zip(*legend_handles.values()),
                      loc='lower center',
                      ncol=5,)

        ax.set_xticks(np.arange(0, image_size[1] + 1, 50), minor=False)
        ax.set_yticks(np.arange(0, image_size[0] + 1, 50), minor=False)
        ax.set_xticks(np.arange(0, image_size[1] + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, image_size[0] + 1, 1), minor=True)
        ax.grid(True, 'major', 'both',
                linestyle='-',
                color='k',
                linewidth=0.035,
                alpha=0.6)
        ax.grid(True, 'minor', 'both',
                linestyle='-',
                color='k',
                linewidth=0.02,
                alpha=0.25)

        csutils.despine(ax, bottom=True, right=True)

        ax.set_xlim([0, image_size[1]])
        ax.set_ylim([0, image_size[0]])
        ax.margins(0.1)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.spines['top'].set_position(('data', -5))
        ax.spines['left'].set_position(('data', -5))

        ax.set_title(figure_name)

        plt.setp(ax.get_xticklabels(), rotation=-90, ha='center')

        print('Saving Figure')
        csutils.save_figures(
            directory=figure_dir,
            filename=figure_savename)

        if csutils.isnotebook():
            print('Displaying preview figure in notebook output:')
            plt.show()
        else:
            plt.close(f)

    except Exception as e:
        print(f'Preview of command file failed with error: {e}')


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

    def extend(self, _iter):
        self.chunks.extend(_iter)

    def append(self, item):
        self.chunks.append(item)

    def pam_command(self,
                    label_to_id=None,
                    auto_assign_chunk_labels=None,
                    split_values=False,
                    build_up_values=False,
                    command_join_str=','):

        if split_values or build_up_values:
            assert split_values ^ build_up_values, \
                ('Only one of `split_values` and `build_up_values` can be set '
                 'to True.')

        if auto_assign_chunk_labels is None:
            auto_assign_chunk_labels = (False
                                        if (split_values or build_up_values)
                                        else True)

        if auto_assign_chunk_labels:
            label_to_id = dict() if label_to_id is None else label_to_id
            next_id_int = 1
            for c in self:
                if c.value not in label_to_id:
                    label_to_id_vals = label_to_id.values()
                    while next_id_int in label_to_id_vals:
                        next_id_int += 1
                    label_to_id[c.value] = next_id_int

                c.label = c.value
            print('Auto-assigned chunk labels to be chunk values, and mapped '
                  'labels to integer ids.')
            print()

        next_call_kwargs = {'label_to_id': label_to_id}

        if split_values or build_up_values:
            print(
                f'Creating a series of ChunkLists based on image vals and '
                f'using method '
                f'{"`split_values`" if split_values else "`build_up_values`"}.')
            chunk_list_map = dict()
            for c in self:
                try:
                    chunk_list_map[c.value].append(c)
                except KeyError:
                    new_list = ChunkList()
                    chunk_list_map[c.value] = new_list
                    chunk_list_map[c.value].append(c)

            chunklist_call_kwargs = next_call_kwargs
            chunklist_call_kwargs['auto_assign_chunk_labels'] = False

            output_dict = dict()
            if split_values:
                _iterator = enumerate(chunk_list_map.items())
                for i, (val, chunklist) in _iterator:
                    chunklist.label = val
                    command = chunklist.pam_command(**chunklist_call_kwargs)

                    if round(val) == val:
                        valstr = f'{val:04d}'
                    else:
                        valstr = f'{val:.3f}'

                    output_dict[f'v_layer={i+1:03d}_value={valstr:s}'] = command

            elif build_up_values:
                values = list(sorted(chunk_list_map.keys()))
                deltas = np.diff([0] + values)

                if auto_assign_chunk_labels:
                    new_label_to_id = [(d, (i + 1))
                                       for i, d in enumerate(deltas)]
                    print('Will overwrite previously auto-assigned labels with '
                          'delta value-based labels [(delta, id_int), ...]:')
                    pp.pprint(new_label_to_id)

                for i, d in enumerate(deltas):
                    chunklist = ChunkList(label=d)
                    for v in values[i:]:
                        chunklist.extend(chunk_list_map[v])

                    if auto_assign_chunk_labels:
                        chunklist_call_kwargs['label_to_id'] = dict(
                            [new_label_to_id[i]])

                    command = chunklist.pam_command(**chunklist_call_kwargs)

                    if round(d) == d:
                        valstr = f'{d:+04d}'
                    else:
                        valstr = f'{d:+.3f}'

                    output_dict[f'd_layer={i+1:03d}_delta={valstr}'] = command

            print('Multi ChunkList command generation complete.\n')
            return output_dict

        else:
            chunk_call_kwargs = next_call_kwargs
            if not auto_assign_chunk_labels:
                chunk_call_kwargs['label'] = self.label

            print('> Generating ChunkList command string.')
            return command_join_str.join(c.pam_command(**chunk_call_kwargs)
                                         for c in self)


class Chunk(object):
    inset_subpixel_width = 1e-2

    def __init__(self, xy_loc, xy_size, total_xy_size, value=1, label=None):
        self.xy_loc = xy_loc
        self.xy_size = xy_size
        self.total_xy_size = total_xy_size
        self.value = value
        self.label = label

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

    def pam_command(self,
                    label=None,
                    label_to_id=None,
                    number_join_str=','):
        label = self.label if label is None else label

        id_int = (1 if label is None or label_to_id is None
                  else label_to_id[label])

        id_int_str = f'{(-1 * id_int):d}'

        return number_join_str.join(['%.*f' % (self._n_decimals, x)
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
    filename = 'test_2d_smile_grey.png'
    im_path = os.path.join('data', 'examples', filename)
    mask = Mask.from_image(im_path)
    mask.write_command_files()


if __name__ == '__main__':
    main()
