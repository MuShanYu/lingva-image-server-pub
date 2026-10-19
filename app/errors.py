"""接口错误：响应体统一为 {"code": ..., "error": ..., <附加字段>}。

code 是稳定的机器可读标识，客户端据此分支处理、显示本地化文案；error 是英文说明，便于调试，
措辞可能调整，客户端不要依赖它。附加字段是文案里要用到的参数（比如 max_mb），客户端翻译时代入。
新增 code 只增不改：已发布的 code 含义不变，README 的错误码表同步更新。
"""

from enum import StrEnum


class Code(StrEnum):
    # 通用
    INVALID_REQUEST = "invalid_request"  # 400 参数缺失或格式不对，附 fields
    NOT_FOUND = "not_found"  # 404 路径不存在
    METHOD_NOT_ALLOWED = "method_not_allowed"  # 405
    INTERNAL_ERROR = "internal_error"  # 500 未预期的异常

    # 鉴权
    UNAUTHORIZED = "unauthorized"  # 401 token 无效或缺失

    # 识别
    INVALID_IMAGE = "invalid_image"  # 400 图片无法解码或格式不支持
    IMAGE_TOO_LARGE = "image_too_large"  # 413 附 max_mb
    INFERENCE_FAILED = "inference_failed"  # 500


class ApiError(Exception):
    def __init__(self, status_code: int, code: Code, message: str, extra: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra or {}  # 除 code、error 外附加到响应体里的字段


def error_body(code: Code, message: str, extra: dict | None = None) -> dict:
    return {"code": str(code), "error": message, **(extra or {})}
