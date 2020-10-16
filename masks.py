#!python3
# masks.py

import numpy as np


class Mask:

    def __init__(self, mask_data):
        self.mask_data = mask_data

    def get_chunks(self):
        chunks = []

        for z_loc in range(self.mask_data.shape[2]):
            locs = np.where(self.mask_data[:, :, z_loc])
            chunk_sublist = []
            for x_loc, y_loc in zip(locs):
                new_chunk = Chunk(
                    xy_loc=(x_loc, y_loc),
                    z_loc=z_loc,
                    xy_size=(1, 1),
                    value=self.mask_data[x_loc, y_loc, z_loc])
                chunk_sublist.append(new_chunk)
            chunks.append(chunk_sublist)

        return chunks

    @property
    def mask_data(self):
        return self._mask_data

    @mask_data.setter
    def mask_data(self, x):
        x_np = np.array(x)
        if x_np.ndim == 2:
            self._mask_data = x_np.reshape((*x_np.shape, -1))
        elif x_np.ndim == 3:
            self._mask_data = x_np
        else:
            raise NotImplementedError

class Chunk:
    def __init__(self, xy_loc, z_loc, xy_size, value, label):
        self.z_loc = z_loc
        self.xy_loc = xy_loc
        self.xy_size = xy_size
        self.value = value
        self.label = label