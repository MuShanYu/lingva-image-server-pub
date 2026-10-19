import io

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class DecodeError(Exception):
    pass


def decode_image(data: bytes) -> Image.Image:
    """解码为按 EXIF 方向转正后的 RGB 图像。"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise DecodeError(f"Unsupported image format: {im.format}, only jpg/png/webp are supported")
            return ImageOps.exif_transpose(im).convert("RGB")
    except DecodeError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as e:
        raise DecodeError("Unable to decode image") from e
