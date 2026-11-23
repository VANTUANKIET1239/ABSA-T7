from contextlib import asynccontextmanager
import logging
import secrets
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
try:
    from .config import Settings
    from .model_service import ModelService, ModelUnavailable
    from .preprocessing import PreprocessingError
    from .schemas import AnalyzeRequest, AnalyzeResponse, summarize
except ImportError:
    from config import Settings
    from model_service import ModelService, ModelUnavailable
    from preprocessing import PreprocessingError
    from schemas import AnalyzeRequest, AnalyzeResponse, summarize

logger = logging.getLogger("absa.api")


def create_app(settings=None, service=None):
    settings = settings or Settings.from_env()
    service = service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(application):
        application.state.startup_error = None
        try:
            await run_in_threadpool(service.load)
            logger.info("ABSA ready on %s; checkpoint %s", service.device, service.version)
        except Exception as exc:
            # No request content is logged and no fake predictions are substituted.
            application.state.startup_error = str(exc) if isinstance(exc, (ModelUnavailable, PreprocessingError)) else "Không nạp được model. Kiểm tra thư viện và cấu hình local."
            logger.error("Model startup failed (%s). See /health.", type(exc).__name__)
        yield

    app = FastAPI(title="Shopee Review Analyzer – ABSA API", version="1.4.0", lifespan=lifespan)
    app.state.model_service = service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"chrome-extension://[a-p]{32}",
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"], allow_credentials=False)

    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize(key: str | None = Depends(key_header)):
        if settings.api_key and not secrets.compare_digest((key or "").encode("utf-8"), settings.api_key.encode("utf-8")):
            raise HTTPException(401, "Khóa API không hợp lệ.")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default error object includes the raw input; omit it.
        return JSONResponse(status_code=422, content={"detail": "reviews phải là danh sách chuỗi, không nhận trường bổ sung."})

    @app.get("/health")
    def health():
        return {"status": "ok" if service.loaded else "unavailable", "model_loaded": service.loaded,
                "device": service.device, "model_version": service.version,
                "batch_size": settings.batch_size, "max_reviews": settings.max_reviews,
                "max_review_chars": settings.max_review_chars,
                "error": getattr(app.state, "startup_error", None)}

    @app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(authorize)])
    async def analyze(payload: AnalyzeRequest):
        reviews = payload.reviews
        if len(reviews) > settings.max_reviews:
            raise HTTPException(413, f"Tối đa {settings.max_reviews} review mỗi request.")
        if any(len(text) > settings.max_review_chars for text in reviews):
            raise HTTPException(413, f"Mỗi review tối đa {settings.max_review_chars} ký tự.")
        if sum(map(len, reviews)) > settings.max_total_chars:
            raise HTTPException(413, f"Tổng nội dung tối đa {settings.max_total_chars} ký tự.")
        if not service.loaded:
            raise HTTPException(503, getattr(app.state, "startup_error", None) or "Model chưa sẵn sàng; kiểm tra /health.")
        try:
            results = await run_in_threadpool(service.analyze, reviews) if reviews else []
        except (ModelUnavailable, PreprocessingError) as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            logger.error("Inference failed (%s)", type(exc).__name__)
            raise HTTPException(500, "Không phân tích được batch. Nội dung review không được ghi vào log.") from exc
        return AnalyzeResponse(total_reviews=len(reviews), results=results, summary=summarize(results),
                               model_version=service.version, device=service.device)

    return app


app = create_app()
