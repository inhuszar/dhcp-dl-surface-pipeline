import torch

from ddsp import resource
from ddsp.seg.unet import UNet


def load(device):
    model = UNet(
        C_in=1,
        C_hid=[16, 32, 64, 128, 128],
        C_out=1,
    ).to(device)
    model.load_state_dict(
        torch.load(resource("seg/model/model_seg.pt"), map_location=device))
    return model
