import torch
from types import SimpleNamespace

from ddsp import resource
from ddsp.surface.net import MeshDeform


def _surface_reconstruction(device, weights):
    model = MeshDeform(
        C_hid=[8, 16, 32, 64, 128, 128],
        C_in=1,
        sigma=1.0,
        interpolation='tricubic',
        device=device,
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    return model


def load(device):
    return SimpleNamespace(
        left=SimpleNamespace(
            wm=_surface_reconstruction(
                device,
                resource('surface/model/model_hemi-left_wm.pt')),
            pial=_surface_reconstruction(
                device,
                resource('surface/model/model_hemi-left_pial.pt')),
        ),
        right=SimpleNamespace(
            wm=_surface_reconstruction(
                device,
                resource('surface/model/model_hemi-right_wm.pt')),
            pial=_surface_reconstruction(
                device,
                resource('surface/model/model_hemi-right_pial.pt')),
        ),
    )
