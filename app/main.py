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
from .schemas import AnalyzeResponse, ErrorResponse, HealthResponse
from .stages.decode import DecodeError

API_VERSION = 1

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
        )

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
        too_large = ApiError(
            413, Code.IMAGE_TOO_LARGE, f"Image exceeds {settings.max_upload_mb} MB", {"max_mb": settings.max_upload_mb}
        )
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise too_large
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise too_large

        pipeline: Pipeline = request.app.state.pipeline
        async with request.app.state.semaphore:
            try:
                result = await run_in_threadpool(pipeline.run, data, inpaint=inpaint, debug=debug)
            except DecodeError as e:
                raise ApiError(400, Code.INVALID_IMAGE, str(e)) from e
            except Exception as e:
                logger.exception("推理失败：%s", file.filename)
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

    return app
