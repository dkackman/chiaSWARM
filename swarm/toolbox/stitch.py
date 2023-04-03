import math
import requests
from PIL import Image, ImageDraw, ImageFont
import io
from ..diffusion.output_processor import make_result, make_thumbnail, image_to_buffer

def stitch_callback(device_id, model_name, **kwargs):
    pipeline_config = {}

    jobs = kwargs["jobs"]
    image_urls = [job["resultUri"] for job in jobs]
    images = download_images(image_urls)
    resized_images = resize_images(images, jobs)
    stitched_image = stitch_images(resized_images)

    buffer = image_to_buffer(stitched_image, "image/jpeg", "web_low")
    thumbnail = make_thumbnail(buffer)
    results = {}
    results["primary"] = make_result(buffer, thumbnail, "image/jpeg")
    return results, pipeline_config


def download_images(image_urls):
    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        images.append(image)
    return images

def resize_images(images, jobs, size=(128, 128)):
    resized_images = []
    for index, image in enumerate(images):
        resized_image = image.resize(size)
        draw = ImageDraw.Draw(resized_image)
        font = ImageFont.truetype("arial.ttf", 16)
        job_index = str(index + 1)
        draw.text((10, 10), job_index, font=font, fill=(255, 255, 255))
        resized_images.append(resized_image)
    return resized_images

def stitch_images(resized_images):
    images_per_row = math.ceil(math.sqrt(len(resized_images)))
    stitched_image_size = 128 * images_per_row
    stitched_image = Image.new('RGB', (stitched_image_size, stitched_image_size))

    x_offset, y_offset = 0, 0
    for img in resized_images:
        stitched_image.paste(img, (x_offset, y_offset))
        x_offset += 128
        if x_offset >= stitched_image_size:
            x_offset = 0
            y_offset += 128

    return stitched_image
