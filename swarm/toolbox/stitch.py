import math
from PIL import Image, ImageDraw
from ..post_processors.output_processor import (
    make_result,
    make_thumbnail,
    image_to_buffer,
)

thumb_size = 144


def stitch_callback(device_id, model_name, **kwargs):
    print("Stitching...")
    pipeline_config = {
        "model_name": model_name,
    }

    jobs = kwargs["jobs"]
    images = kwargs["images"]
    resized_images = resize_images(images)
    stitched_image = stitch_images(resized_images)

    _buffer = image_to_buffer(stitched_image, "image/jpeg", "web_high")
    thumbnail = make_thumbnail(_buffer)
    results = {"primary": make_result(_buffer, thumbnail, "image/jpeg")}
    image_map = generate_image_map(resized_images, jobs)
    pipeline_config["image_map"] = image_map
    return results, pipeline_config


def resize_images(images, size=(thumb_size, thumb_size)):
    resized_images = []
    for index, image in enumerate(images):
        width, height = image.size
        aspect_ratio = float(width) / float(height)

        # Calculate new dimensions while preserving aspect ratio
        if width > height:
            new_width = min(size[0], width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(size[1], height)
            new_width = int(new_height * aspect_ratio)

        # Ensure the new dimensions do not exceed the thumbnail size
        new_width = min(new_width, size[0])
        new_height = min(new_height, size[1])

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        draw = ImageDraw.Draw(resized_image)
        job_index = str(index + 1)
        draw.text((10, 10), job_index, fill=(255, 255, 255))
        resized_images.append(resized_image)

    return resized_images


def stitch_images(resized_images):
    images_per_row = math.ceil(math.sqrt(len(resized_images)))
    stitched_image_size = thumb_size * images_per_row
    stitched_image = Image.new("RGB", (stitched_image_size, stitched_image_size))

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
        coords = (
            f"{x_offset},{y_offset},{x_offset + thumb_size},{y_offset + thumb_size}"
        )
        href = jobs[index]["resultUri"]
        file_name = jobs[index].pop("fileName", href)
        alt = jobs[index].pop("model_name", f"Image {index + 1}")

        map_data.append(
            {
                "shape": "rect",
                "coords": coords,
                "href": href,
                "alt": alt,
                "filename": file_name,
            }
        )

        x_offset += thumb_size
        if x_offset >= thumb_size * math.ceil(math.sqrt(len(resized_images))):
            x_offset = 0
            y_offset += thumb_size

    return map_data
