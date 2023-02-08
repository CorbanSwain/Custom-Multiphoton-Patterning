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
import matplotlib as mpl
import pprint as pp
import enum


log = csutils.get_logger(__name__)
csutils.apply_standard_logging_config(window_format='cli')

class Mask(object):
    save_dir_tag = 'mask_outputs'

    def __init__(self, mask_data=None, load_path=None):
        self.mask_data = mask_data
        self.load_path = load_path
        self._meta_str = ''
        self._did_pixel_edge_shift = False

        self._save_dir = None

    def create_chunk_lists(self,
                           do_points_only=False,
                           dim_priority='x',
                           do_group_chunks=True):
        log.info('Chunking mask file into rectangular regions ...')

        chunk_lists = []

        z_size, tot_y_size, tot_x_size = self.mask_data.shape

        for z_loc, zslice in enumerate(self.mask_data):
            log.info('> Analyzing mask at z-slice {:3d} of {:3d}',
                     z_loc + 1, z_size)

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
                        log.info('> > Analyzing mask at {:s} {:6,d}',
                                 'row' if dim_priority == 0 else 'column',
                                 a_loc + 1)
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

        log.info('Done chunking.')
        log.info('Mask decomposed into {:,d} chunks.',
                 sum([len(cl.chunks) for cl in chunk_lists]))
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

        log.info('Loaded image data has shape: {}', mask_data.shape)

        return cls(mask_data=mask_data, load_path=im_path)

    def shift_pixel_edges(self,
                          rising_edge_shift,
                          falling_edge_shift,
                          scan_axis=-1,
                          do_plot_results=True,
                          **kwargs):

        if self._did_pixel_edge_shift:
            log.warning('Pixel edge shifting is being performed multiple'
                        ' times, further modifying the mask data; this is'
                        ' not recommend, and may cause unexpected results.'
                        ' Consider first reloading the mask, then applying'
                        ' pixel edge shifts with this class method or manually'
                        ' applying pixel edge shifts using the module function'
                        ' prior to mask creation.')

        self.mask_data = shift_pixel_edges(
            self.mask_data,
            scan_axis=scan_axis,
            rising_edge_shift=rising_edge_shift,
            falling_edge_shift=falling_edge_shift,
            do_plot_results=do_plot_results,
            **kwargs)

        self._meta_str += (f'rise_shft={rising_edge_shift:+d}_'
                           f'fall_shft={falling_edge_shift:+d}_')
        self._did_pixel_edge_shift = True



    @csutils.timed(log)
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

        log.info('Creating photoactivation mask file(s) for import into '
                 'Prairie View.')

        chunk_list_kwargs = chunk_list_kwargs or dict()
        pam_command_kwargs = pam_command_kwargs or dict()

        if do_optimize_file_size:
            log.info('\nOptimizing file size by testing different chunking '
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
                log.info('METHOD {:d} of {:d}: do_invert={}, '
                         'dim_priority="{}"',
                         i + 1,
                         n_trials,
                         do_invert,
                         dim_priority)
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
                log.info('\n')

            log.info('BEST SIZE found to be {:,d} chunks with method # {:d}.',
                     best_size, best_method + 1)

        else:
            chunk_lists = self.create_chunk_lists(**chunk_list_kwargs)

        size_str = 'size={1:d}x{0:d}'.format(*self.mask_data.shape[1:])

        if save_dir is None:
            save_dir = self.save_dir

        csutils.touchdir(save_dir)
        make_path = ft.partial(os.path.join, save_dir)

        log.info('Will save mask file(s) in directory "{:s}".\n', save_dir)

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
                log.info('Digitizing `Chunk` values to have only '
                         '{:d} levels', max_num_output_values)
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
                log.warning('overwriting `label_to_id` parameter in '
                            '`pam_command_kwargs`.')

            pam_command_kwargs['label_to_id'] = label_to_id

            log.info('Auto-set `label_to_id` dict based on values of entire '
                     'mask.')

            log.info('Writing file with conversion factors between palette '
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

            log.info('Wrote lookup table to "{:s}":', file_path)
            log.info(''.join(firstlines + lines))
            log.info('\n')
        else:
            # TODO - Handle max output values when not auto-setting label to id.
            #        Maybe need to pass param to the `cl.pam_command()`
            #        function.
            log.warning(
                'WARNING: **Not Implemented**, `max_num_output_values` '
                'parameter will be ignored since `auto_set_label_to_id` is '
                'False.')

        file_paths = []

        for i, cl in enumerate(chunk_lists):
            cmd_like = cl.pam_command(**pam_command_kwargs)

            z_index = i + 1

            if type(cmd_like) is dict:
                if do_prepare_sequential_import:
                    log.warning(
                        'NOT IMPLEMENTED: Cannot prepare sequential import.')
                    do_prepare_sequential_import = False

                for title, cmd_str in cmd_like.items():
                    file_name = (f'{size_str}_z={z_index:03d}_{invert_flag}'
                                 f'{self._meta_str}'
                                 f'{title}_pam.txt')
                    file_path = make_path(file_name)
                    log.debug(file_path)
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
                file_name = (f'{size_str}_z={z_index:03d}_{invert_flag}'
                             f'{self._meta_str}pam.txt')
                file_path = make_path(file_name)
                log.info(file_path)
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
            log.info(
                '\nCopying PAM files to subfolder to prepare for sequential '
                'z-stack import.')

            seq_dir = make_path(f'SEQ_IMPORT_{size_str}x{len(file_paths)}_'
                                f'{invert_flag}{self._meta_str}pams')
            log.info('Sequential import directory: "{:s}"', seq_dir)
            if os.path.exists(seq_dir):
                shutil.rmtree(seq_dir)
            csutils.touchdir(seq_dir)
            for z_index, file_path in file_paths:
                new_path = os.path.join(seq_dir, f'{z_index:03d}.txt')
                shutil.copyfile(file_path, new_path)

        log.info('\nMask command file(s) generation complete!')


class RegionType(enum.IntEnum):
    NONE = 0
    START_PLATEAU = enum.auto()
    END_PLATEAU = enum.auto()
    START_WELL = enum.auto()
    END_WELL = enum.auto()
    CENTER_PLATEAU = enum.auto()
    CENTER_WELL = enum.auto()
    CENTER_STEP = enum.auto()

    @classmethod
    def to_width_cutoff(cls, type_arr, rising_edge_shift, falling_edge_shift):
        out_arr = np.zeros(type_arr.shape, dtype=int)
        out_arr[type_arr == cls.START_PLATEAU] = -falling_edge_shift
        out_arr[type_arr == cls.END_PLATEAU] = rising_edge_shift
        out_arr[type_arr == cls.START_WELL] = -rising_edge_shift
        out_arr[type_arr == cls.END_WELL] = falling_edge_shift
        out_arr[type_arr == cls.CENTER_PLATEAU] = (rising_edge_shift
                                                   - falling_edge_shift)
        out_arr[type_arr == cls.CENTER_WELL] = (falling_edge_shift
                                                - rising_edge_shift)
        return out_arr


def _start_pos_to_end_pos(start_pos,
                          break_locs,
                          line_length):
    # compute the end (not inclusive) position of each region
    # by shifting the start position vector
    end_pos = start_pos.copy()
    end_pos[break_locs] = line_length
    return np.append(end_pos[1:], line_length)


def shift_pixel_edges(x,
                      scan_axis=-1,
                      *,
                      rising_edge_shift,
                      falling_edge_shift,
                      warn=True,
                      reverse_scan_direction=False,
                      do_plot_results=True):

    if (rising_edge_shift == 0) and (falling_edge_shift == 0):
        return x

    positive_expansion = falling_edge_shift - rising_edge_shift

    if positive_expansion > 0:
        trans_type_str = 'EXPAND'
    elif positive_expansion < 0:
        trans_type_str = 'CONTRACT'
        positive_expansion = -positive_expansion
    else:
        trans_type_str = 'TRANSLATE'
        positive_expansion = falling_edge_shift

    log.info('Shifting the edges of the data along scan_axis {:d}, the'
             ' transformation will {:s} positive-valued regions by'
             ' approx {:d} pixels, shifting rising edges {:s} by {:d} pixels'
             ' and falling edges {:s} by {:d} pixels.',
             scan_axis,
             trans_type_str,
             positive_expansion,
             'forward' if rising_edge_shift >= 0 else 'backward',
             abs(rising_edge_shift),
             'forward' if falling_edge_shift >= 0 else 'backward',
             abs(falling_edge_shift))

    # move scan axis to the final axis
    x = np.moveaxis(x, scan_axis, -1)
    moveax_shape = x.shape
    scanline_length = moveax_shape[-1]

    # reshape to have scan axis along axis 1 and all other dims along axis 0
    x = x.reshape((-1, scanline_length))
    if do_plot_results:
        x_compare = x
    all_lines_shape = x.shape
    num_scanlines = all_lines_shape[0]

    # flip processed array if using a reversed scan direction
    # i.e. scanning from scanline position -1 to position 0
    if reverse_scan_direction:
        x = np.fliplr(x)

    log.debug('Shifting pixel edges for {:d} scanlines,'
              ' each scanline has length {:d}.',
              num_scanlines, scanline_length)

    # use np.diff() along the scan direction to determine where all changes
    # in value occur. Note that lines with no changes in value will not be
    # considered in the following computations
    diffs = np.diff(x, axis=-1)
    edge_index_tup = np.where(diffs != 0)
    raw_edge_line_ind, raw_edge_pos = edge_index_tup

    # indexes where scanline breaks occur in the raw_edge_pos vector
    (line_break_locs_raw,) = np.where(
        np.diff(raw_edge_line_ind, prepend=-1) != 0)

    # convert from the raw edge position to a vector of every "left" edge
    # where either a pixel value changes or a scanline begins
    region_start_pos = raw_edge_pos + 1
    region_start_pos = np.insert(region_start_pos,
                                 line_break_locs_raw,
                                 0)
    # number of identified regions
    num_regions = region_start_pos.size
    log.debug('Identified {:d} regions across all hetrogenously'
              '-valued scanlines.', num_regions)

    # the vector index of the beginnin of each scanline break
    (line_break_locs,) = np.where(region_start_pos == 0)

    # the index of the line corrresponding to each region
    region_line_ind = np.insert(raw_edge_line_ind,
                                line_break_locs_raw,
                                raw_edge_line_ind[line_break_locs_raw])

    # extract the pixel value for each region
    region_vals = x[region_line_ind, region_start_pos]

    if warn:
        region_end_pos = _start_pos_to_end_pos(
            region_start_pos, line_break_locs, scanline_length)
        region_widths = region_end_pos - region_start_pos

        line_start_locs = line_break_locs
        line_end_locs = np.append(line_break_locs[1:] - 1,
                                  num_regions - 1)

        line_start_filt = np.zeros((num_regions,), dtype=bool)
        line_start_filt[line_start_locs] = True
        line_end_filt = np.zeros((num_regions,), dtype=bool)
        line_end_filt[line_end_locs] = True

        end_diff = np.diff(region_vals, n=1, prepend=0)
        start_diff = np.diff(region_vals, n=1, append=0)
        ddiff = start_diff - end_diff

        region_class = np.zeros((num_regions,), dtype=RegionType)
        region_class[ddiff == 0] = RegionType.CENTER_STEP
        region_class[ddiff > 0] = RegionType.CENTER_WELL
        region_class[ddiff < 0] = RegionType.CENTER_PLATEAU
        region_class[line_end_filt
                     & (end_diff > 0)] = RegionType.END_PLATEAU
        region_class[line_end_filt
                     & (end_diff < 0)] = RegionType.END_WELL
        region_class[line_start_filt
                     & (start_diff > 0)] = RegionType.START_WELL
        region_class[line_start_filt
                     & (start_diff < 0)] = RegionType.START_PLATEAU

        width_cutoffs = RegionType.to_width_cutoff(
            region_class,
            rising_edge_shift=rising_edge_shift,
            falling_edge_shift=falling_edge_shift)
        lost_regions = region_widths <= width_cutoffs

        if np.any(lost_regions):
            num_lost_regions = np.sum(lost_regions)
            log.warning(
                '{:d} scanline region(s) will be lost when performing pixel'
                ' shifts because the region\'s width is too narrow.',
                num_lost_regions)

            fail_start_pos = region_start_pos[lost_regions]
            fail_line_ind = region_line_ind[lost_regions]
            fail_region_width = region_widths[lost_regions]

            plt.figure(figsize=(10, 10))
            plt.imshow(x, cmap='gray')
            highlights = [
                mpl.patches.Rectangle((stt - 0.5, l - 0.5), w, 1)
                for stt, l, w in zip(fail_start_pos,
                                     fail_line_ind,
                                     fail_region_width)]
            patch_props = dict(
                facecolor='r',
                alpha=0.4,
                lw=1,
                edgecolor='r',
                hatch='x')
            hlpc = mpl.collections.PatchCollection(
                highlights, **patch_props)
            p = mpl.patches.Patch(**patch_props,
                                  label='lost regions')

            default_viewport = 75
            y_lims = [
                np.max([np.min(fail_line_ind) - (default_viewport / 2), -0.5]),
                np.min([np.max(fail_line_ind)
                        + (default_viewport / 2), num_scanlines + 0.5])]
            x_lims = [
                np.max([np.min(fail_start_pos) - (default_viewport / 2), -0.5]),
                np.min([np.max(fail_start_pos + fail_region_width)
                        + (default_viewport / 2), num_scanlines + 0.5])]
            if reverse_scan_direction:
                x_lims = np.flip(np.array(x_lims))
                x_label_addn = ', Reversed'
            else:
                x_label_addn = ''

            plt.xlim(x_lims)
            plt.ylim(np.flip(np.array(y_lims)))
            plt.xlabel(f'Scanline Position{x_label_addn} (pixel)')
            plt.ylabel('Scanline Index (a.u.)')
            plt.title('WARNING: The highlighted region(s) will'
                      ' be lost\nwhen performing pixel edge shifts.',
                      color='r', fontweight='bold')
            plt.legend(handles=[p],
                       bbox_to_anchor=(1.02, 1.00),
                       loc='upper left')
            ax = plt.gca()
            ax.add_collection(hlpc)
            plt.show()

    # compute the change from value to value across regions to classify the
    # edge type
    region_val_diff = np.diff(region_vals, prepend=0)

    # mask for rising edge, falling, edge, and positions which can be
    # validly shifted
    is_pos_rising = region_val_diff > 0
    is_pos_falling = np.logical_not(is_pos_rising)
    is_pos_shiftable = np.ones((num_regions,), dtype=bool)
    is_pos_shiftable[line_break_locs] = False

    # If the position is determined to be a rising edge AND can be
    # validly shifted, then shift by the parametrized amount. Similarly,
    # for the falling edge.
    new_region_start_pos = region_start_pos.copy()
    rise_filt = is_pos_rising & is_pos_shiftable
    new_region_start_pos[rise_filt] += rising_edge_shift
    fall_filt = np.logical_not(is_pos_rising) & is_pos_shiftable
    new_region_start_pos[fall_filt] += falling_edge_shift
    new_region_start_pos = np.clip(new_region_start_pos, 0, scanline_length)

    # new end position is computed
    new_region_end_pos = _start_pos_to_end_pos(
        new_region_start_pos, line_break_locs, scanline_length)

    # compute the new region widths after perfoming shifts
    new_region_widths = new_region_end_pos - new_region_start_pos

    # subset each of the following variabled to only retain values where
    # the modified region has a positive width
    valid_regions = new_region_widths > 0
    v_region_line_ind = region_line_ind[valid_regions]
    v_region_start_pos = new_region_start_pos[valid_regions]
    v_region_vals = region_vals[valid_regions]
    v_region_widths = new_region_widths[valid_regions]

    # convert the line-index and scanline-position-index
    # into linear indexes
    raveled_region_start_pos = np.ravel_multi_index(
        (v_region_line_ind, v_region_start_pos),
        all_lines_shape)
    raveled_region_end_pos = (raveled_region_start_pos
                              + v_region_widths)

    # the following lines implement a vectorized np.arange() with two vectors
    # of corresponding starts and stops and the output assembled into a
    # single concatenated vector of the results
    repeat_lengths = raveled_region_end_pos - raveled_region_start_pos
    vector_arange = (
            np.repeat(raveled_region_end_pos - repeat_lengths.cumsum(),
                      repeat_lengths)
            + np.arange(repeat_lengths.sum()))
    # region values are repeated to match the raveled and aranged indexes
    updated_values = np.repeat(v_region_vals, repeat_lengths)

    # finally we will update the input ndarray with the new regions
    x_prime = x.copy().ravel()
    x_prime[vector_arange] = updated_values

    # reshape and transform the output to that of the original input
    x_prime = x_prime.reshape((-1, scanline_length))
    if reverse_scan_direction:
        x_prime = np.fliplr(x_prime)

    if do_plot_results:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.imshow(x_prime, cmap='gray')
        im_diff = x_prime.astype(float) - x_compare.astype(int)
        im_diff[im_diff < 0] = -1
        im_diff[im_diff > 0] = 1
        is_diff = im_diff != 0
        cmap = mpl.colormaps['PiYG']
        ax.imshow(im_diff,
                  cmap=cmap,
                  alpha=is_diff.astype(float),
                  vmin=-1,
                  vmax=1)
        ax.set_xlabel('Scanline Position (pixel)')
        ax.set_ylabel('Scanline Index (a.u.)')
        ax.set_title(f'Result of pixel edge shift with changes highlighted\n'
                     f'(rising_edge_shift = {rising_edge_shift:+d},'
                     f' falling_edge_shift = {falling_edge_shift:+d})',
                     fontweight='bold')
        p0 = mpl.patches.Patch(color=cmap(0.0), label='removed\nor decreased')
        p1 = mpl.patches.Patch(color=cmap(1.0), label='added\nor increased')
        ax.legend(handles=[p0, p1],
                  title='Changed Regions',
                  framealpha=1,
                  bbox_to_anchor=(1.02, 1.00),
                  loc='upper left')
        plt.show()

    x_prime = x_prime.reshape(moveax_shape)
    return np.moveaxis(x_prime, -1, scan_axis)




def split_file(pth, max_size, preview_shape=None):
    full_file_size = os.path.getsize(pth)
    num_outfiles = ceil(full_file_size / max_size)

    max_size_mb = max_size / 1e6

    if num_outfiles > 1:

        log.info('Splitting output file into {:d} files of approx {:.2f} MB.\n',
                 num_outfiles, max_size_mb)

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
        log.info(
            'No file splitting performed since the input file is under the '
            '{:.2f} MB limit.',
            max_size_mb)


def preview_command_file(file_path,
                         image_size=(1, 1),
                         verbose=False):
    try:
        log.info('\nAttempting to preview command file "{:s}"', file_path)

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
            log.info('Insufficient number of values ({:d}) found in '
                     'command file to produce any mask shapes.',
                     len(tokens))
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
                    log.info('Chunk {:6,d} : {:s}', i + 1, str(c))

            log.info('Creating Figure')
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

        log.info('Saving Figure')
        csutils.save_figures(
            directory=figure_dir,
            filename=figure_savename)

        if csutils.isnotebook():
            log.info('Displaying preview figure in notebook output:')
            plt.show()
        else:
            plt.close(f)

    except Exception as e:
        log.warning('Preview of command file failed with error.', exc_info=e)


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
            log.info(
                'Auto-assigned chunk labels to be chunk values, and mapped '
                'labels to integer ids.')
            log.info('')

        next_call_kwargs = {'label_to_id': label_to_id}

        if split_values or build_up_values:
            log.info(
                'Creating a series of ChunkLists based on image vals and '
                'using method {:s}.',
                "`split_values`" if split_values else "`build_up_values`")
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
                    log.info('Will overwrite previously auto-assigned labels '
                             'with delta value-based labels [(delta, id_int), '
                             '...]:')
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

            log.info('Multi ChunkList command generation complete.\n')
            return output_dict

        else:
            chunk_call_kwargs = next_call_kwargs
            if not auto_assign_chunk_labels:
                chunk_call_kwargs['label'] = self.label

            log.info('> Generating ChunkList command string.')
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
