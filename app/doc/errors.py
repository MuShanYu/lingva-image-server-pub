"""文档识别的异常。不依赖第三方库：前端（Dockerfile.front，不装 numpy）的远程流水线也要用。"""


class DocDisabled(Exception):
    """服务器没有启用文档识别（DOC_PIPELINE_ENABLED=0）。"""


class DocTimeout(Exception):
    """单页处理超过 DOC_TIMEOUT。"""


class DocInferenceError(Exception):
    """worker 识别时出错，或 worker 进程异常退出。"""


class ImageTooLarge(Exception):
    """图片长边超过 DOC_MAX_SIDE。"""

    def __init__(self, max_side: int) -> None:
        super().__init__(f"Image exceeds {max_side} px on the long side")
        self.max_side = max_side
