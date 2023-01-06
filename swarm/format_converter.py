import io


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
