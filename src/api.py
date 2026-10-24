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
from .tools import pipeline as tools_pipeline
from .ratelimit import RateLimiter


providers: Dict[str, BaseProvider] = {}

# 每平台控频：同平台请求强制最小间隔（防上游风控/禁言），见 config.yaml rate_limit
rate_limiter = RateLimiter(
    default_interval_ms=settings.rate_limit.min_interval_ms,
    per_provider_ms=settings.rate_limit.overrides,
)


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
            providers[provider_name] = KimiProvider(
                token=token,
                refresh_token=settings.providers.kimi.refresh_token,
            )
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
            providers[provider_name] = KimiProvider(
                token=token,
                refresh_token=settings.providers.kimi.refresh_token,
            )
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
    version="1.5.0",
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
    return {"message": "NXAPI - OpenAI Compatible API", "version": "1.5.0"}


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    models = []
    
    if settings.providers.deepseek.token:
        provider = DeepSeekProvider(token=settings.providers.deepseek.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="deepseek"))
    
    if settings.providers.kimi.token:
        provider = KimiProvider(
            token=settings.providers.kimi.token,
            refresh_token=settings.providers.kimi.refresh_token,
        )
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="moonshot"))
    
    if settings.providers.metaso.token:
        provider = MetasoProvider(token=settings.providers.metaso.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="metaso"))
    
    if settings.providers.doubao.token:
        provider = DoubaoProvider(token=settings.providers.doubao.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="doubao"))
    
    if settings.providers.qwen.token:
        provider = QwenProvider(token=settings.providers.qwen.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="qwen"))
    
    if settings.providers.zhipu.token:
        provider = ZhipuProvider(token=settings.providers.zhipu.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="zhipu"))
    
    if settings.providers.minimax.token:
        provider = MiniMaxProvider(token=settings.providers.minimax.token)
        for model_id in provider.models:
            models.append(ModelInfo(id=model_id, owned_by="minimax"))
    
    return ModelList(data=models)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    # ---- R1：不传 tools 时完全走原路径（旧逻辑一行未改，结构性保证 I1）----
    if not request.tools:
        # C2：content 放宽为可选后，在非 tools 路径显式拦截，避免语义缝隙
        if any(message.content is None for message in request.messages):
            raise HTTPException(
                status_code=400,
                detail="content is required when tools are not provided",
            )

        try:
            provider_name, provider = get_provider_for_model(request.model)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        await rate_limiter.acquire(provider_name)

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

        try:
            response = await provider.chat_completion(request)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # I1：剥离本次新增字段，保持非 tools 响应与改造前逐字段等价
        payload = response.model_dump()
        message = payload["choices"][0]["message"]
        message.pop("tool_calls", None)
        message.pop("tool_call_id", None)
        return JSONResponse(payload)

    # ---- 工具调用路径（方案 A：提示词模拟）----
    # R4：role=tool 必须携带 tool_call_id
    if any(message.role == "tool" and not message.tool_call_id for message in request.messages):
        raise HTTPException(
            status_code=400,
            detail="tool_call_id is required when role is tool",
        )

    try:
        provider_name, provider = get_provider_for_model(request.model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    await rate_limiter.acquire(provider_name)

    prepared = tools_pipeline.prepare(request)

    if request.stream:
        return StreamingResponse(
            tools_pipeline.stream(prepared, provider, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    try:
        upstream = await provider.chat_completion(prepared.request)
        upstream_text = upstream.choices[0].message.content or "" if upstream.choices else ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return tools_pipeline.build_response(prepared, upstream_text)


async def stream_chat_completion(
    provider: BaseProvider,
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    try:
        async for chunk in provider.chat_completion_stream(request):
            data = chunk.model_dump_json(exclude_none=True)
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        # 流已开始后无法改 HTTP 状态码；部分客户端（如 Octop）不解析
        # {"error": ...} 裸错误帧，改用标准 chunk 信封把错误文本作为内容发出
        data = ChatCompletionChunk(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta=DeltaMessage(content=f"[上游错误] {e}"),
                finish_reason="stop",
            )]
        ).model_dump_json(exclude_none=True)
        yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"


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


# ---- /v2 透传端点：纯净对话通道 ----

def to_passthrough_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """丢弃 system、忽略 tools，把剩余消息原样发给上游。

    首条 user 消息完全不加前缀；多轮时用「用户：/助手：」最小拼接——
    因为每次请求都会在上游新建会话，不拼历史就没有上下文。
    """
    parts: List[str] = []
    for m in request.messages:
        if not m.content:
            continue
        if m.role == "user":
            parts.append(m.content if not parts else f"用户：{m.content}")
        elif m.role == "assistant":
            parts.append(f"助手：{m.content}")
        elif m.role == "tool":
            parts.append(f"工具结果：{m.content}")
        # system 一律丢弃：透传通道不携带客户端注入的任何提示词
    if not parts:
        raise HTTPException(status_code=400, detail="v2 透传通道没有可发送的消息内容")

    return request.model_copy(update={
        "messages": [ChatMessage(role="user", content="\n\n".join(parts))],
        "tools": None,
        "tool_choice": None,
    })


@app.post("/v2/chat/completions")
async def chat_completions_passthrough(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None)
):
    return await chat_completions(to_passthrough_request(request), authorization)


@app.get("/v2/models", response_model=ModelList)
async def list_models_passthrough():
    return await list_models()


if __name__ == "__main__":
    run_server()
