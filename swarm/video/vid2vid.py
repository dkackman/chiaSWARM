import cv2
import requests
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
from moviepy.editor import *
from ..diffusion.output_processor import make_result
from io import BytesIO
import tempfile
import pathlib


def model_video_callback(device_id, model_name, **kwargs):
    pipeline_config = {}
    results = {}

    pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)  # type: ignore
    pipeline.enable_attention_slicing()  # type: ignore
    pipeline.enable_xformers_memory_efficient_attention()  # type: ignore
    pipeline.unet.to(memory_format=torch.channels_last)  # type: ignore
    pipeline = pipeline.to(f"cuda:{device_id}")  # type: ignore

    prompt = kwargs["prompt"]
    negative_prompt = kwargs.pop("negative_prompt", "")
    guidance_scale = kwargs.pop("guidance_scale", 7.5)
    image_guidance_scale = kwargs.pop("image_guidance_scale", 1.5)
    num_inference_steps = kwargs.pop("num_inference_steps", 15)

    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = pathlib.Path(tmpdirname)
        video_file_path = download_video(temp_dir, kwargs.pop("video_uri"))
        break_vid = get_frames(temp_dir, video_file_path)
        stride = 1
        frames_list = break_vid[0]
        fps = break_vid[1]
        n_frame = int(len(frames_list) / stride)
        result_frames = []

        nsfw_content_detected = False
        for frame in frames_list[0::stride]:
            print(f"{frame} of {n_frame}")

            pix2pix_img = img2img(
                pipeline,
                prompt,
                guidance_scale,
                image_guidance_scale,
                frame,
                num_inference_steps,
                negative_prompt,
                512,
                512,
            )
            images = pix2pix_img[0]
            nsfw_content_detected = pix2pix_img[1] or nsfw_content_detected
            rgb_im = images[0].convert("RGB")

            # exporting the image
            file_path = str(temp_dir.joinpath(f"{frame}.vid2vid.jpg"))
            rgb_im.save(file_path)
            result_frames.append(file_path)

        final_filepath = create_video(temp_dir, result_frames, fps / stride)
        with open(final_filepath, "rb") as video_file:
            video_buffer = BytesIO(video_file.read())

        with open(result_frames[0], "rb") as thumb_file:
            thumb_buffer = BytesIO(thumb_file.read())

        results["primary"] = make_result(video_buffer, thumb_buffer, "video/mp4")
        pipeline_config["nsfw"] = nsfw_content_detected
        pipeline_config["cost"] = 512 * 512 * num_inference_steps * len(result_frames)

        return results, pipeline_config


def download_video(tmpdir, video_uri):
    head = requests.head(video_uri, allow_redirects=True)
    content_length = head.headers.pop("Content-Length", 0)
    content_type = head.headers.pop("Content-Type", "")

    if not content_type.startswith("video"):
        raise Exception(
            f"Input does not appear to be an video.\nContent type was {content_type}."
        )

    # to protect worker nodes, no external videos over 30 MiB
    if int(content_length) > 1048576 * 30:
        raise Exception(
            f"Input video too large.\nMax size is {1048576 * 30} bytes.\nImage was {content_length}."
        )

    file_path = str(tmpdir.joinpath("input.mp4"))
    response = requests.get(video_uri, allow_redirects=True, stream=True)
    if response.ok:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print(
            "Download failed: status code {}\n{}".format(
                response.status_code, response.text
            )
        )

    return file_path


def img2img(
    pipeline,
    prompt,
    text_guidance_scale,
    image_guidance_scale,
    image,
    steps,
    neg_prompt,
    width,
    height,
):
    image = Image.open(image)
    ratio = min(height / image.height, width / image.width)
    image = image.resize(
        (int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS
    )

    result = pipeline(
        prompt,
        negative_prompt=neg_prompt,
        image=image,
        num_inference_steps=int(steps),
        image_guidance_scale=image_guidance_scale,
        guidance_scale=text_guidance_scale,
    )

    return result.images, result.nsfw_content_detected[0]


def get_frames(temp_dir, video_in):
    frames = []
    # resize the video
    clip = VideoFileClip(video_in)

    resized_video_path = str(temp_dir.joinpath("video_resized.mp4"))
    # check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)  # type: ignore
        clip_resized.write_videofile(resized_video_path, fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)  # type: ignore
        clip_resized.write_videofile(resized_video_path, fps=clip.fps)

    print("video resized to 512 height")

    # Opens the Video file with CV2
    cap = cv2.VideoCapture(resized_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        image_path = str(temp_dir.joinpath(f"kang{i}.jpg"))
        cv2.imwrite(image_path, frame)
        frames.append(image_path)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames, fps


def create_video(temp_dir, frames, fps):
    video_path = str(temp_dir.joinpath("output.mp4"))
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(video_path, fps=fps)

    return video_path
