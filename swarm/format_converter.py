import io


def image_to_buffer(image, content_type):
    if content_type.startswith("image"):
        format = "PNG" if content_type == "image/png" else "JPEG"
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        buffer.seek(0)
        return buffer

    raise NotImplementedError(content_type)
