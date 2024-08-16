#!/usr/bin/env python

import numpy as np

import ants
import torch
import nibabel as nib
from types import SimpleNamespace

from ddsp import resource
from ddsp.volume.volume import Volume


def load_t2_atlas():
    return Volume(
        antsvol=ants.image_read(
            resource("template/dhcp_week-40_template_T2w.nii.gz")
        ),
        niftiheader=nib.load(
            resource("template/dhcp_week-40_template_T2w.nii.gz")
        ).header,
        mask=None,
    )


def load_surface_template():
    return SimpleNamespace(
        left=nib.load(
            resource("template/dhcp_week-40_hemi-left_init.surf.gii")
        ),
        right=nib.load(
            resource("template/dhcp_week-40_hemi-right_init.surf.gii")
        ),
    )


# ------ load input sphere ------
def load_input_sphere(device):
    # Left hemisphere
    sphere_left_in = nib.load(
        resource("template/dhcp_week-40_hemi-left_sphere.surf.gii")
    )
    vert_sphere_left_in = sphere_left_in.agg_data("pointset")
    vert_sphere_left_in = torch.Tensor(vert_sphere_left_in[None]).to(device)

    # Right hemisphere
    sphere_right_in = nib.load(
        resource("template/dhcp_week-40_hemi-right_sphere.surf.gii")
    )
    vert_sphere_right_in = sphere_right_in.agg_data("pointset")
    vert_sphere_right_in = torch.Tensor(vert_sphere_right_in[None]).to(device)

    return SimpleNamespace(
        left=vert_sphere_left_in,
        right=vert_sphere_right_in,
    )


# ------ load template sphere (160k) ------
def load_template_sphere(device):
    sphere_160k = nib.load(resource("template/sphere_163842.surf.gii"))
    vert_sphere_160k = sphere_160k.agg_data("pointset")
    face_160k = sphere_160k.agg_data("triangle")
    vert_sphere_160k = torch.Tensor(vert_sphere_160k[None]).to(device)
    face_160k = torch.LongTensor(face_160k[None]).to(device)
    return SimpleNamespace(
        vertices=vert_sphere_160k,
        faces=face_160k,
    )


def load_barycentric_coordinates():
    """
    Load pre-computed barycentric coordinates for spherical interpolation.

    """
    # Left hemisphere
    barycentric_left = nib.load(
        resource("template/dhcp_week-40_hemi-left_barycentric.gii")
    )
    bc_coord_left = barycentric_left.agg_data("pointset")
    face_left_id = barycentric_left.agg_data("triangle")

    # Right hemisphere
    barycentric_right = nib.load(
        resource("template/dhcp_week-40_hemi-right_barycentric.gii")
    )
    bc_coord_right = barycentric_right.agg_data("pointset")
    face_right_id = barycentric_right.agg_data("triangle")

    return SimpleNamespace(
        vertices=SimpleNamespace(
            left=bc_coord_left,
            right=bc_coord_right,
        ),
        faces=SimpleNamespace(
            left=face_left_id,
            right=face_right_id,
        ),
    )
