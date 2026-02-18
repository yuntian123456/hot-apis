import json
import time
import uuid
from typing import Optional, List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import settings
from .models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatMessage, ChatCompletionChoice, StreamChoice, DeltaMessage, Usage,
    ModelList, ModelInfo
)
from .providers import DeepSeekProvider, KimiProvider, MetasoProvider, DoubaoProvider, QwenProvider, ZhipuProvider, MiniMaxProvider, BaseProvider


providers: Dict[str, BaseProvider] = {}


def get_provider_for_model(model: str) -> tuple[str, BaseProvider]:
    model_lower = model.lower()
    
    if any(x in model_lower for x in ["deepseek", "ds-"]):
        provider_name = "deepseek"
        if provider_name not in providers:
            token = settings.providers.deepseek.token
            providers[provider_name] = DeepSeekProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["kimi", "moonshot"]):
        provider_name = "kimi"
        if provider_name not in providers:
            token = settings.providers.kimi.token
            providers[provider_name] = KimiProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["metaso"]):
        provider_name = "metaso"
        if provider_name not in providers:
            token = settings.providers.metaso.token
            providers[provider_name] = MetasoProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["doubao"]):
        provider_name = "doubao"
        if provider_name not in providers:
            token = settings.providers.doubao.token
            providers[provider_name] = DoubaoProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["qwen", "tongyi"]):
        provider_name = "qwen"
        if provider_name not in providers:
            token = settings.providers.qwen.token
            providers[provider_name] = QwenProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["zhipu", "chatglm", "glm"]):
        provider_name = "zhipu"
        if provider_name not in providers:
            token = settings.providers.zhipu.token
            providers[provider_name] = ZhipuProvider(token=token)
        return provider_name, providers[provider_name]
    
    if any(x in model_lower for x in ["minimax"]):
        provider_name = "minimax"
        if provider_name not in providers:
            token = settings.providers.minimax.token
            providers[provider_name] = MiniMaxProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.deepseek.token:
        provider_name = "deepseek"
        if provider_name not in providers:
            token = settings.providers.deepseek.token
            providers[provider_name] = DeepSeekProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.kimi.token:
        provider_name = "kimi"
        if provider_name not in providers:
            token = settings.providers.kimi.token
            providers[provider_name] = KimiProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.metaso.token:
        provider_name = "metaso"
        if provider_name not in providers:
            token = settings.providers.metaso.token
            providers[provider_name] = MetasoProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.doubao.token:
        provider_name = "doubao"
        if provider_name not in providers:
            token = settings.providers.doubao.token
            providers[provider_name] = DoubaoProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.qwen.token:
        provider_name = "qwen"
        if provider_name not in providers:
            token = settings.providers.qwen.token
            providers[provider_name] = QwenProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.zhipu.token:
        provider_name = "zhipu"
        if provider_name not in providers:
            token = settings.providers.zhipu.token
            providers[provider_name] = ZhipuProvider(token=token)
        return provider_name, providers[provider_name]
    
    if settings.providers.minimax.token:
        provider_name = "minimax"
        if provider_name not in providers:
            token = settings.providers.minimax.token
            providers[provider_name] = MiniMaxProvider(token=token)
        return provider_name, providers[provider_name]
    
    raise HTTPException(status_code=400, detail=f"No provider available for model: {model}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for provider in providers.values():
        await provider.close()


app = FastAPI(
    title="NXAPI - OpenAI Compatible API",
    description="大模型 API 中转站，支持 DeepSeek、Kimi、Metaso、豆包、千问、智谱清言和 MiniMax",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "NXAPI - OpenAI Compatible API", "version": "1.0.0"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    models = []
    
    if settings.providers.deepseek.token:
        models.extend([
            ModelInfo(id="deepseek-chat", owned_by="deepseek"),
            ModelInfo(id="deepseek-reasoner", owned_by="deepseek"),
            ModelInfo(id="deepseek-r1", owned_by="deepseek"),
        ])
    
    if settings.providers.kimi.token:
        models.extend([
            ModelInfo(id="kimi", owned_by="moonshot"),
            ModelInfo(id="kimi-k2.5", owned_by="moonshot"),
            ModelInfo(id="moonshot-v1-8k", owned_by="moonshot"),
            ModelInfo(id="moonshot-v1-32k", owned_by="moonshot"),
            ModelInfo(id="moonshot-v1-128k", owned_by="moonshot"),
        ])
    
    if settings.providers.metaso.token:
        models.extend([
            ModelInfo(id="metaso", owned_by="metaso"),
            ModelInfo(id="metaso-concise", owned_by="metaso"),
            ModelInfo(id="metaso-detail", owned_by="metaso"),
            ModelInfo(id="metaso-research", owned_by="metaso"),
            ModelInfo(id="metaso-concise-scholar", owned_by="metaso"),
            ModelInfo(id="metaso-detail-scholar", owned_by="metaso"),
            ModelInfo(id="metaso-research-scholar", owned_by="metaso"),
        ])
    
    if settings.providers.doubao.token:
        models.extend([
            ModelInfo(id="doubao", owned_by="doubao"),
            ModelInfo(id="doubao-pro", owned_by="doubao"),
            ModelInfo(id="doubao-lite", owned_by="doubao"),
            ModelInfo(id="doubao-1.5-pro", owned_by="doubao"),
            ModelInfo(id="doubao-1.5-lite", owned_by="doubao"),
        ])
    
    if settings.providers.qwen.token:
        models.extend([
            ModelInfo(id="qwen", owned_by="qwen"),
            ModelInfo(id="qwen3", owned_by="qwen"),
            ModelInfo(id="qwen3.5-plus", owned_by="qwen"),
            ModelInfo(id="qwen3-max", owned_by="qwen"),
            ModelInfo(id="qwen3-max-thinking", owned_by="qwen"),
            ModelInfo(id="qwen3-flash", owned_by="qwen"),
            ModelInfo(id="qwen3-coder", owned_by="qwen"),
            ModelInfo(id="qwen-vl-plus", owned_by="qwen"),
            ModelInfo(id="qwen-vl-max", owned_by="qwen"),
            ModelInfo(id="qwen-long", owned_by="qwen"),
        ])
    
    if settings.providers.zhipu.token:
        models.extend([
            ModelInfo(id="zhipu", owned_by="zhipu"),
            ModelInfo(id="chatglm", owned_by="zhipu"),
            ModelInfo(id="glm-5", owned_by="zhipu"),
        ])
    
    if settings.providers.minimax.token:
        models.extend([
            ModelInfo(id="minimax", owned_by="minimax"),
            ModelInfo(id="minimax-auto", owned_by="minimax"),
            ModelInfo(id="MiniMax-M2.5", owned_by="minimax"),
        ])
    
    return ModelList(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    try:
        provider_name, provider = get_provider_for_model(request.model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(provider, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        try:
            response = await provider.chat_completion(request)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


async def stream_chat_completion(
    provider: BaseProvider,
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in provider.chat_completion_stream(request):
            data = chunk.model_dump_json(exclude_unset=True, exclude_none=True)
            yield f"data: {data}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_data = json.dumps({"error": {"message": str(e), "type": "internal_error"}})
        yield f"data: {error_data}\n\n"


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "api_error"}}
    )


def run_server():
    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port
    )


if __name__ == "__main__":
    run_server()
