import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .errors import ApiError, Code, error_body
from .pipeline import Pipeline, load_pipeline
from .schemas import AnalyzeResponse, DocFigureResponse, DocPageResponse, ErrorResponse, HealthResponse
from .stages.decode import DecodeError

API_VERSION = 1
# lang_hint：语种代码（zh、ja、ko、ru、th、pt-BR 等），只做格式校验，未知语种按默认识别模型处理
_LANG_HINT = r"^[A-Za-z]{0,8}([_-][A-Za-z0-9]{1,8})*$"
_DOC_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}

logger = logging.getLogger("lingva")


def _error(status_code: int, code: Code, message: str, extra: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_body(code, message, extra))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pipeline = load_pipeline(settings)
        app.state.pipeline = pipeline
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrency)
        # 文档 worker 一次只处理一个请求：先在这里排队，拿到后再占用与漫画共用的推理名额，
        # 免得排队中的文档请求占着名额挡住漫画请求。排队优先级、每台设备的并发上限由前端处理
        app.state.doc_semaphore = asyncio.Semaphore(1)
        logger.info(
            "流水线已就绪：stub=%s device=%s max_concurrency=%d models=%s",
            pipeline.stub,
            pipeline.device,
            settings.max_concurrency,
            pipeline.model_version,
        )
        yield

    app = FastAPI(title="Lingva image analysis", version=str(API_VERSION), lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message, exc.extra)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 框架自己抛的：路径不存在、方法不对等
        code = {404: Code.NOT_FOUND, 405: Code.METHOD_NOT_ALLOWED}.get(exc.status_code, Code.INVALID_REQUEST)
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()]
        return _error(400, Code.INVALID_REQUEST, f"Invalid request parameters: {', '.join(fields)}", {"fields": fields})

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # 未预期的异常也按统一格式返回，而不是框架默认的纯文本 500；堆栈由框架重新抛出后交给 uvicorn 记录
        return _error(500, Code.INTERNAL_ERROR, "Internal server error")

    bearer = HTTPBearer(auto_error=False)

    def require_token(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if cred is None or not secrets.compare_digest(
            cred.credentials.encode(), settings.api_token.encode()
        ):
            raise ApiError(401, Code.UNAUTHORIZED, "Invalid or missing token")

    @app.get("/v1/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        pipeline: Pipeline = request.app.state.pipeline
        return HealthResponse(
            api_version=API_VERSION,
            stub=pipeline.stub,
            device=pipeline.device,
            model_version=pipeline.model_version,
            capabilities=pipeline.capabilities,
            doc_pipeline=pipeline.doc_pipeline,
        )

    async def _read_upload(file: UploadFile) -> bytes:
        too_large = ApiError(
            413, Code.IMAGE_TOO_LARGE, f"Image exceeds {settings.max_upload_mb} MB", {"max_mb": settings.max_upload_mb}
        )
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise too_large
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise too_large
        return data

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        dependencies=[Depends(require_token)],
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def analyze(
        request: Request,
        file: UploadFile = File(...),
        inpaint: bool = Form(True),
        debug: bool = Form(False),
    ) -> AnalyzeResponse:
        data = await _read_upload(file)
        pipeline: Pipeline = request.app.state.pipeline
        async with request.app.state.semaphore:
            try:
                result = await run_in_threadpool(pipeline.run, data, inpaint=inpaint, debug=debug)
            except DecodeError as e:
                raise ApiError(400, Code.INVALID_IMAGE, str(e)) from e
            except Exception as e:
                # 不记文件名（客户端提供，可能含用户信息）
                logger.exception("推理失败（%d 字节）", len(data))
                raise ApiError(500, Code.INFERENCE_FAILED, "Inference failed") from e

        logger.info(
            "analyze %dx%d blocks=%d inpaint=%s debug=%s timings_ms=%s",
            result.width,
            result.height,
            len(result.blocks),
            inpaint,
            debug,
            result.timings_ms,
        )
        return result

    async def _doc_route(request: Request, file: UploadFile, mode: str, lang_hint: str | None, erase: bool) -> dict:
        from .doc.errors import DocDisabled, DocInferenceError, DocTimeout, ImageTooLarge

        pipeline: Pipeline = request.app.state.pipeline
        disabled = ApiError(503, Code.DOC_PIPELINE_DISABLED, "Document recognition is not enabled on this server")
        if not pipeline.doc_pipeline.enabled:
            raise disabled
        data = await _read_upload(file)
        async with request.app.state.doc_semaphore, request.app.state.semaphore:
            try:
                result = await run_in_threadpool(pipeline.run_doc, mode, data, lang_hint=lang_hint or None, erase=erase)
            except DecodeError as e:
                raise ApiError(400, Code.INVALID_IMAGE, str(e)) from e
            except ImageTooLarge as e:
                raise ApiError(
                    413, Code.IMAGE_TOO_LARGE, str(e), {"max_mb": settings.max_upload_mb, "max_side": e.max_side}
                ) from e
            except DocDisabled as e:
                raise disabled from e
            except DocTimeout as e:
                raise ApiError(504, Code.TIMEOUT, "Processing timed out") from e
            except DocInferenceError as e:
                logger.error("文档识别失败（%s，%d 字节）：%s", mode, len(data), e)
                raise ApiError(500, Code.INFERENCE_FAILED, "Inference failed") from e
            except Exception as e:
                logger.exception("文档识别失败（%s，%d 字节）", mode, len(data))
                raise ApiError(500, Code.INFERENCE_FAILED, "Inference failed") from e

        items = result.get("blocks") if mode == "doc_page" else result.get("regions")
        # 只记元数据：识别出的文字和文件名不进日志
        logger.info(
            "%s lang=%s blocks=%d erase=%s timings_ms=%s", mode, result.get("lang"), len(items or []), erase, result.get("timing_ms")
        )
        return result

    @app.post(
        "/v1/doc/page", response_model=DocPageResponse, dependencies=[Depends(require_token)], responses=_DOC_ERROR_RESPONSES
    )
    async def doc_page(
        request: Request,
        file: UploadFile = File(...),
        lang_hint: str | None = Form(None, max_length=32, pattern=_LANG_HINT),
        erase: bool = Form(False),
    ) -> dict:
        return await _doc_route(request, file, "doc_page", lang_hint, erase)

    @app.post(
        "/v1/doc/figure", response_model=DocFigureResponse, dependencies=[Depends(require_token)], responses=_DOC_ERROR_RESPONSES
    )
    async def doc_figure(
        request: Request,
        file: UploadFile = File(...),
        lang_hint: str | None = Form(None, max_length=32, pattern=_LANG_HINT),
        erase: bool = Form(False),
    ) -> dict:
        return await _doc_route(request, file, "doc_figure", lang_hint, erase)

    return app
