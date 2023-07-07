import numpy as np
import os
from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
import cv2

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import tempfile
from PIL import Image


def gfpgan_process(img_list):
    bg_upsampler = None
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

    only_center_face = False
    aligned = False
    weight = 0.5
    upscale = 1
    ext = "png"
    suffix = None

    arch = "clean"
    channel_multiplier = 2
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    model_path = url
    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
    )
    outputs = []
    for pil_image in img_list:
        numpy_array = np.array(pil_image)
        input_img = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        basename = "gpgan"
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            for idx, (cropped_face, restored_face) in enumerate(
                zip(cropped_faces, restored_faces)
            ):
                # save cropped face
                save_crop_path = os.path.join(
                    tmpdirname, "cropped_faces", f"{basename}_{idx:02d}.png"
                )
                imwrite(cropped_face, save_crop_path)
                # save restored face
                if suffix is not None:
                    save_face_name = f"{basename}_{idx:02d}_{suffix}.png"
                else:
                    save_face_name = f"{basename}_{idx:02d}.png"
                save_restore_path = os.path.join(
                    tmpdirname, "restored_faces", save_face_name
                )
                imwrite(restored_face, save_restore_path)
                # save comparison image
                cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
                imwrite(
                    cmp_img,
                    os.path.join(tmpdirname, "cmp", f"{basename}_{idx:02d}.png"),
                )

            # save restored img
            if restored_img is not None:
                if ext == "auto":
                    extension = ext[1:]
                else:
                    extension = ext

                if suffix is not None:
                    save_restore_path = os.path.join(
                        tmpdirname,
                        "restored_imgs",
                        f"{basename}_{suffix}.{extension}",
                    )
                else:
                    save_restore_path = os.path.join(
                        tmpdirname, "restored_imgs", f"{basename}.{extension}"
                    )
                # imwrite(restored_img, save_restore_path)

                outputs.append(Image.fromarray(restored_img))

    return outputs
