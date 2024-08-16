#!/usr/bin/env python

import numpy as np

import ants
import torch
import nibabel as nib

from ddsp import resource
from ddsp.volume.volume import Volume


def load_t2_atlas():
    return Volume(
        antsvol=ants.image_read(
            resource('template/dhcp_week-40_template_T2w.nii.gz')),
        niftiheader=nib.load(
            resource('template/dhcp_week-40_template_T2w.nii.gz')).header,
        mask=None,
    )


def load_input_surface(device):

    # Left
    surf_left_in = nib.load(
        resource('template/dhcp_week-40_hemi-left_init.surf.gii'))

    # Right
    surf_right_in = nib.load(
        resource('template/dhcp_week-40_hemi-right_init.surf.gii'))
    vert_right_in = surf_right_in.agg_data('pointset')
    face_right_in = surf_right_in.agg_data('triangle')
    vert_right_in = apply_affine_mat(
        vert_right_in, np.linalg.inv(affine_t2_atlas))
    face_right_in = face_right_in[:, [2, 1, 0]]
    vert_right_in = torch.Tensor(vert_right_in[None]).to(device)
    face_right_in = torch.LongTensor(face_right_in[None]).to(device)

# ------ load input sphere ------
def load_input_sphere(device):
    sphere_left_in = nib.load(
        resource('template/dhcp_week-40_hemi-left_sphere.surf.gii'))
    vert_sphere_left_in = sphere_left_in.agg_data('pointset')
    vert_sphere_left_in = torch.Tensor(vert_sphere_left_in[None]).to(device)

    sphere_right_in = nib.load(
        resource('template/dhcp_week-40_hemi-right_sphere.surf.gii'))
    vert_sphere_right_in = sphere_right_in.agg_data('pointset')
    vert_sphere_right_in = torch.Tensor(vert_sphere_right_in[None]).to(device)


# ------ load template sphere (160k) ------
def load_template_sphere(device):
    sphere_160k = nib.load(resource('template/sphere_163842.surf.gii'))
    vert_sphere_160k = sphere_160k.agg_data('pointset')
    face_160k = sphere_160k.agg_data('triangle')
    vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
    face_160k = torch.LongTensor(face_160k[None]).to(device)

# ------ load pre-computed barycentric coordinates ------
# for spherical interpolation
barycentric_left = nib.load(
    resource('template/dhcp_week-40_hemi-left_barycentric.gii'))
bc_coord_left = barycentric_left.agg_data('pointset')
face_left_id = barycentric_left.agg_data('triangle')

barycentric_right = nib.load(
    resource('template/dhcp_week-40_hemi-right_barycentric.gii'))
bc_coord_right = barycentric_right.agg_data('pointset')
face_right_id = barycentric_right.agg_data('triangle')