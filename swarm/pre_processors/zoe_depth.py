import torch
import matplotlib
import matplotlib.cm
import numpy as np
from PIL import Image


def load_zoe():
    torch.hub.help(
        "intel-isl/MiDaS", "DPT_BEiT_L_384"
    )  # Triggers fresh download of MiDaS repo
    model_zoe_n = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval()
    return model_zoe_n.to("cuda")


def colorize(
    value,
    vmin=None,
    vmax=None,
    cmap="gray_r",
    invalid_val=-99,
    invalid_mask=None,
    background_color=(128, 128, 128, 255),
    gamma_corrected=False,
    value_transform=None,
):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    # gamma correction
    img = img / 255
    img = np.power(img, 2.2)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    return img
