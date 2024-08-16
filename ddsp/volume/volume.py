#!/usr/bin/env python

# IMPORTS

import ants
import numpy as np
import nibabel as nib


# IMPLEMENTATION

class Volume(object):
    """
    Image volume container that aims to simplify the syntax of manipulating
    image data across multiple formats (ANTs, NumPy).

    """
    def __init__(
            self,
            antsvol: ants.ANTsImage,
            niftiheader: nib.Nifti1Image or nib.Nifti2Header,
            mask: np.ndarray = None
    ):
        super(Volume, self).__init__()
        self._antsvol = antsvol
        self.header = niftiheader
        self.mask = mask

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, hdr):
        if isinstance(hdr, (nib.Nifti1Header, nib.Nifti2Header)):
            self._header = hdr
        else:
            raise TypeError(
                f"Expected Nifti1Header or Nifti2Header, "
                f"got {type(hdr)} instead.")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        if m is None:
            self._mask = None
        elif hasattr(m, '__array__'):
            self._mask = np.asanyarray(m)
        else:
            raise TypeError('Mask volume must be a NumPy array.')

    def antsvol(self, data=None, apply_mask=True):
        apply_mask = apply_mask and (self.mask is not None)
        if (data is None) and not apply_mask:
            return self._antsvol

        data = data or self.data()
        if apply_mask:
            data = data * self.mask

        return ants.from_numpy(
            data,
            origin=self._antsvol.origin,
            spacing=self._antsvol.spacing,
            direction=self._antsvol.direction,
            has_components=self._antsvol.has_components,
            is_rgb=self._antsvol.is_rgb,
        )

    def affine(self):
        return self.header.get_best_affine()

    def data(self):
        return self._antsvol.numpy()

