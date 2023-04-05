import math
import requests
from PIL import Image, ImageDraw
import io
import json
from ..output_processor import make_result, make_thumbnail, image_to_buffer

thumb_size = 144
def stitch_callback(device_id, model_name, **kwargs):
    print("Stitching...")
    pipeline_config = {
        "model_name": model_name,
    }

    jobs = kwargs["jobs"]
    image_urls = [job["resultUri"] for job in jobs]
    images = download_images(image_urls)
    resized_images = resize_images(images)
    stitched_image = stitch_images(resized_images)

    buffer = image_to_buffer(stitched_image, "image/jpeg", "web_high")
    thumbnail = make_thumbnail(buffer)
    results = {}
    results["primary"] = make_result(buffer, thumbnail, "image/jpeg")
    image_map = generate_image_map(resized_images, jobs)
    pipeline_config["image_map"] = image_map
    return results, pipeline_config


def download_images(image_urls):
    images = []
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        images.append(image)
    return images

def resize_images(images, size=(thumb_size, thumb_size)):
    resized_images = []
    for index, image in enumerate(images):
        resized_image = image.resize(size)
        draw = ImageDraw.Draw(resized_image)
        job_index = str(index + 1)
        draw.text((10, 10), job_index, fill=(255, 255, 255))
        resized_images.append(resized_image)
    return resized_images

def stitch_images(resized_images):
    images_per_row = math.ceil(math.sqrt(len(resized_images)))
    stitched_image_size = thumb_size * images_per_row
    stitched_image = Image.new('RGB', (stitched_image_size, stitched_image_size))

    x_offset, y_offset = 0, 0
    for img in resized_images:
        stitched_image.paste(img, (x_offset, y_offset))
        x_offset += thumb_size
        if x_offset >= stitched_image_size:
            x_offset = 0
            y_offset += thumb_size

    return stitched_image

def generate_image_map(resized_images, jobs):
    map_data = []
    x_offset, y_offset = 0, 0
    for index, _ in enumerate(resized_images):
        coords = f"{x_offset},{y_offset},{x_offset + thumb_size},{y_offset + thumb_size}"
        href = jobs[index]["resultUri"]
        fileName = jobs[index]["fileName"]
        area_data = {
            "shape": "rect",
            "coords": coords,
            "href": href,
            "alt": f"Image {index + 1}",
            "filename": fileName
        }
        map_data.append(area_data)

        x_offset += thumb_size
        if x_offset >= thumb_size * math.ceil(math.sqrt(len(resized_images))):
            x_offset = 0
            y_offset += thumb_size

    return map_data
