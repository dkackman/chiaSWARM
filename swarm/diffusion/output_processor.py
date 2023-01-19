from .video_maker import make_video
import hashlib
import io
from PIL import Image, ImageDraw
import base64


class OutputProcessor:
    def __init__(self, output_list, main_content_type):
        self.intermediate_images = []
        self.outputs = []
        self.output_list = output_list
        self.main_content_type = main_content_type

    def need_intermediates(self):
        return (
            "inference_video" in self.output_list
            or "inference_image_strip" in self.output_list
        )

    def add_latents(self, pipeline, latents):
        latents = 1 / 0.18215 * latents
        image = pipeline.vae.decode(latents).sample[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).numpy()
        self.intermediate_images.extend(pipeline.numpy_to_pil(image))

    def add_outputs(self, images):
        self.outputs.extend(images)

    def get_results(self):
        results = {}
        if "primary" in self.output_list:
            primary_result = post_process(self.outputs)
            primary_result_buffer = image_to_buffer(
                primary_result, self.main_content_type
            )
            results["primary"] = make_result(
                primary_result_buffer, self.main_content_type
            )

        if "inference_image_strip" in self.output_list:
            image_strip = image_grid(
                self.intermediate_images, 1, len(self.intermediate_images)
            )
            image_strip_buffer = image_to_buffer(image_strip, "image/jpeg")
            results["inference_image_strip"] = make_result(
                image_strip_buffer, "image/jpeg"
            )

        if "inference_video" in self.output_list:
            video = make_video(self.intermediate_images + [self.outputs[0]], 5)
            video_buffer = image_to_buffer(video, "video/webm")
            results["inference_video"] = make_result(video_buffer, "video/webm")

        return results


def make_result(buffer, content_type):
    thumbnail_buffer = make_thumbnail(buffer)
    return {
        "blob": base64.b64encode(buffer.getvalue()).decode("UTF-8"),
        "content_type": content_type,
        "thumbnail": base64.b64encode(thumbnail_buffer.getvalue()).decode("UTF-8"),
        "sha256_hash": hashlib.sha256(buffer.getvalue()).hexdigest(),
    }


def make_thumbnail(buffer):
    image = Image.open(buffer).convert("RGB")
    image.thumbnail((100, 100), Image.Resampling.LANCZOS)
    return image_to_buffer(image, "image/jpeg", "web_low")


def post_process(image_list) -> Image.Image:
    num_images = len(image_list)
    if num_images == 1:
        image = image_list[0]
    elif num_images == 2:
        image = image_grid(image_list, 1, 2)
    elif num_images <= 4:
        image = image_grid(image_list, 2, 2)
    elif num_images <= 6:
        image = image_grid(image_list, 2, 3)
    elif num_images <= 9:
        image = image_grid(image_list, 3, 3)
    else:
        raise (Exception("too many images"))

    return image


def image_grid(image_list, rows, cols) -> Image.Image:
    w, h = image_list[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(image_list):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


def image_to_buffer(image, content_type, quality="web_high"):
    if content_type.startswith("image"):
        buffer = io.BytesIO()
        if content_type == "image/png":
            image.save(buffer, format="PNG")
        else:
            image.save(
                buffer, format="JPEG", quality=quality, optimize=True, progressive=True
            )

        buffer.seek(0)
        return buffer

    raise NotImplementedError(content_type)
