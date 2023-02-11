import cv2
import requests
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
from moviepy.editor import *
from ..diffusion.output_processor import make_result
from io import BytesIO


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

    video_file_path = download_video(kwargs.pop("video_uri"))
    break_vid = get_frames(video_file_path)
    step = 1
    frames_list = break_vid[0]
    fps = break_vid[1]
    n_frame = int(len(frames_list) / step)
    result_frames = []

    for frame in frames_list[0::step]:
        print(f"{frame} of {n_frame}")

        pix2pix_img = pix2pix(
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
        rgb_im = images[0].convert("RGB")

        # exporting the image
        rgb_im.save(f"/tmp/result_img-{frame}.jpg")
        result_frames.append(f"/tmp/result_img-{frame}.jpg")

    final_filepath = create_video(result_frames, fps / step)
    with open(final_filepath, "rb") as video_file:
        video_buffer = BytesIO(video_file.read())

    with open(result_frames[0], "rb") as thumb_file:
        thumb_buffer = BytesIO(thumb_file.read())

    results["primary"] = make_result(video_buffer, thumb_buffer, "video/mp4")

    return results, pipeline_config


def download_video(video_uri):
    file_path = "/tmp/video.mp4"
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


def pix2pix(
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
    image = Image.open(f"/tmp/{image}")
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

    return result.images, result.nsfw_content_detected


def get_frames(video_in):
    frames = []
    # resize the video
    clip = VideoFileClip(video_in)

    # check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)  # type: ignore
        clip_resized.write_videofile("/tmp/video_resized.mp4", fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)  # type: ignore
        clip_resized.write_videofile("/tmp/video_resized.mp4", fps=clip.fps)

    print("video resized to 512 height")

    # Opens the Video file with CV2
    cap = cv2.VideoCapture("/home/don/video_resized.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(f"/tmp/kang{i}.jpg", frame)
        frames.append(f"kang{i}.jpg")
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    return frames, fps


def create_video(frames, fps):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile("/home/don/movie.mp4", fps=fps)

    return "/home/don/movie.mp4"
