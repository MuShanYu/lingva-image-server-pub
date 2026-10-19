import base64
import io

from PIL import Image


def encode_webp_b64(img: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=4)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")
