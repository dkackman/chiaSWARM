import aiohttp
import asyncio
import qrcode
from io import BytesIO
from PIL import Image, ImageOps
from .pre_processors.controlnet import (
    preprocess_image,
    resize_for_condition_image,
)

max_size = 1024


async def get_image(uri, size):
    if not isNotBlank(uri):
        return None

    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.head(
            uri, allow_redirects=True, timeout=timeout.total
        ) as response:
            response.raise_for_status()

            content_length = response.headers.get("Content-Length", 0)
            content_type = response.headers.get("Content-Type", "")

            if not content_type.startswith("image"):
                raise Exception(
                    f"Input does not appear to be an image.\nContent type was {content_type}."
                )

            # to protect worker nodes, no external images over 3 MiB
            if int(content_length) > 1048576 * 3:
                raise Exception(
                    f"Input image too large.\nMax size is {1048576 * 3} bytes.\nImage was {content_length}."
                )

        async with session.get(uri) as response:
            response.raise_for_status()
            content = await response.read()

            image = Image.open(BytesIO(content))

        image = ImageOps.exif_transpose(image).convert("RGB")

        # if we have a desired size and the image is larger than it, scale the image down
        if size != None and (image.height > size[0] or image.width > size[1]):
            image.thumbnail(size, Image.Resampling.LANCZOS)

        elif image.height > max_size or image.width > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return image


async def get_control_image(start_image, controlnet, size):
    # base the resolution of of size - defaulting to 768
    W, H = size if size is not None else (768, 768)
    resolution = max(H, W)

    # return the image and the pre-processed image if controlnet is specified
    # preprocess means generate the control image from the input image
    if controlnet.get("preprocess", False):
        return preprocess_image(start_image, controlnet, resolution)

    # user specified control image - go get it
    if isNotBlank(controlnet.get("control_image_uri", None)):
        control_image = await get_image(controlnet.get("control_image_uri"), size)
        return resize_for_condition_image(control_image, resolution)

    # user passed a qrcode - generate image
    if isNotBlank(controlnet.get("qr_code_contents", None)):
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(controlnet.get("qr_code_contents"))
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        return resize_for_condition_image(qrcode_image, resolution)

    # control image and input image are the same
    return resize_for_condition_image(start_image, resolution)


async def download_images(image_urls):
    images = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in image_urls:
            task = async_download_image(session, url)
            tasks.append(task)
        images = await asyncio.gather(*tasks)

    return images


async def async_download_image(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        content = await response.read()

    return Image.open(BytesIO(content))


def isNotBlank(myString):
    return bool(myString and myString.strip())
