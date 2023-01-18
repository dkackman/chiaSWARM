from .video_maker import make_video
from PIL import Image


class OutputProcessor:
    def __init__(self, output_list):
        self.intermediate_images = []
        self.outputs = []
        self.output_list = output_list

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

    def get_video(self):
        if "inference_video" in self.output_list:
            make_video(self.intermediate_images + [self.outputs[0]], 5)

    def get_final_image(self):
        if "main_result" in self.output_list:
            return post_process(self.outputs)

    def get_strip(self):
        if "inference_image_strip" in self.output_list:
            return image_grid(
                self.intermediate_images, 1, len(self.intermediate_images)
            )


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
