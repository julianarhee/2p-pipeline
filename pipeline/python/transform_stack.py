#!/usr/bin/env python2

import os
import numpy as np
import tifffile as tf


from PIL import Image


def coord_transform(x, y, z, affine):
    """ Convert the x, y, z coordinates from one image space to another
        space.

        Parameters
        ----------
        x : number or ndarray
            The x coordinates in the input space
        y : number or ndarray
            The y coordinates in the input space
        z : number or ndarray
            The z coordinates in the input space
        affine : 2D 4x4 ndarray
            affine that maps from input to output space.

        Returns
        -------
        x : number or ndarray
            The x coordinates in the output space
        y : number or ndarray
            The y coordinates in the output space
        z : number or ndarray
            The z coordinates in the output space

        Warning: The x, y and z have their Talairach ordering, not 3D
        numy image ordering.
    """
    coords = np.c_[np.atleast_1d(x).flat,
                   np.atleast_1d(y).flat,
                   np.atleast_1d(z).flat,
                   np.ones_like(np.atleast_1d(z).flat)].T
    x, y, z, _ = np.dot(affine, coords)
    return x.squeeze(), y.squeeze(), z.squeeze()


class VAST_SegmentationSections(object):
    def __init__(self, pngs_folder, from_section=0, to_section=None, color_type=0):
        self._pngs_folder = pngs_folder
        self._cached_images = {}
        self._sections_num = None
        self._sections_fnames = None
        self._iter_idx = 0
        self._from_section = from_section
        self._to_section = to_section
        self._color_type = color_type
        if color_type == 1: # random colors
            # Generate a random colormap (should be the same for all nodes)
            np.random.seed(7)
            ncolors = 1000
            self._color_map = np.uint8(np.random.randint(0,256,(ncolors+1)*3)).reshape((ncolors + 1, 3))
            self._color_map[0] = np.array([0, 0, 0])
    
    def _read_folder(self):
        if self._sections_fnames is None:
            self._sections_fnames = sorted(glob.glob(os.path.join(self._pngs_folder, '*.png')))
            if self._to_section is not None:
                self._sections_fnames = self._sections_fnames[:self._to_section]
            if self._from_section > 0:
                self._sections_fnames = self._sections_fnames[self._from_section:]
    
    def size(self):
        if self._sections_num is None:
            self._read_folder()
            self._sections_num = len(self._sections_fnames)
        return self._sections_num
    
    def read_section(self, idx):
        if idx in self._cached_images.keys():
            return self._cached_images[idx]
        
        self._read_folder()
        section_fname = self._sections_fnames[idx]
        rgb_img = cv2.imread(section_fname, 1)
        if self._color_type == 0:
            img = np.zeros((rgb_img.shape[:2]), dtype=np.uint8)
            img[rgb_img[:, :, 0] > 0] = 255
            img[rgb_img[:, :, 1] > 0] = 255
            img[rgb_img[:, :, 2] > 0] = 255
        elif self._color_type == 1: # random color
            img = self.seg_to_color(rgb_img)
        elif self._color_type == 2:
            img = rgb_img
        
        self._cached_images[idx] = img
        
        return self._cached_images[idx]
    
    def all_images(self):
        idx = 0
        while idx < self.size():
            yield self.read_section(idx)
            idx += 1

    @property
    def shape(self):
        img1 = self.read_section(0)
        return (self.size(), img1.shape[0], img1.shape[1])
    
    def seg_to_color(self, ids_img):
        red = ids_img[:, :, 0].flatten().astype(np.uint32)
        green = ids_img[:, :, 1].flatten().astype(np.uint32)
        blue = ids_img[:, :, 2].flatten().astype(np.uint32)
        
        ids = (blue << 16) + (green << 8) + red
        overlay_colors = self._color_map[ids]
        colored_img = overlay_colors.reshape(ids_img.shape)

        return colored_img

        #ids = np.zeros_like(ids_img.shape[:2], dtype=np.uint32)
#        ids_in_img = set(ids) # a set of unique ids
#        new_ids_in_img = ids_in_img - self._color_map.keys()
#        new_ids_color_map = {new_id:np.uint8(np.random.randint(0,256,3)) for new_id in new_ids_in_img}
#        self._color_map.update(new_ids_color_map)
#        colors = self._color_map[]
#        for c in self._color_map:
#            colors[colors == ]
#        colors
        #colors = np.zeros_like(ids_img, dtype=np.uint8)
        #colors[:,:,0] = np.mod(107*ids_img[:, :, 0], 700).astype(np.uint8)
        #colors[:,:,1] = np.mod(509*ids_img[:, :, 1], 900).astype(np.uint8)
        #colors[:,:,2] = np.mod(205*ids_img[:, :, 2], 777).astype(np.uint8)
#        return colors

#aff = (0.98999999, -0.06, -0.07, 2.58000002, 0.07, 1.0, 0.04, -8.45000001)

tiff_sections_path = '/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/em/direct_em7_to_stack_color.tif'

volume = tf.imread(tiff_sections_path)

cells_vast_folder = '/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/em/cells_exported_Richard_20170514/'
cells_vast_sections_color = VAST_SegmentationSections(cells_vast_folder, color_type=1)
colormap = cells_vast_sections_color._color_map

outpath = '/nas/volume1/2photon/RESDATA/phase1_block2/alignment2/em/TEST'
if not os.path.exists(outpath):
    os.mkdir(outpath)


tiff_sections = os.listdir(tiff_sections_path)

for zslice in tiff_sections:
    currslice = Image.open(os.path.join(tiff_sections_path, zslice)).convert('RGB')
    outslice = currslice.transform(currslice.size, Image.AFFINE, aff, resample=Image.NEAREST)
    outslice.save(os.path.join(outpath, zslice))
