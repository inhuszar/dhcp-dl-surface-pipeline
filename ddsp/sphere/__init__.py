import torch
from types import SimpleNamespace

from ddsp import resource
from ddsp.sphere.net.sunet import SphereDeform


def _spherical_projection(device, weights):
    model = SphereDeform(
        C_in=6,
        C_hid=[32, 64, 128, 256, 256],
        device=device,
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    return model


def load(device):

    left = _spherical_projection(
        device, resource("sphere/model/model_hemi-left_sphere.pt")
    )
    right = _spherical_projection(
        device, resource("sphere/model/model_hemi-right_sphere.pt")
    )

    return SimpleNamespace(
        left=left,
        right=right,
    )
