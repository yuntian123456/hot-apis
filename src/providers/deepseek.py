import httpx
import json
import base64
import asyncio
import struct
from typing import Optional, AsyncGenerator, Dict, Any, List
from pathlib import Path
import time
import uuid

from ..models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from ..models import ChatMessage, ChatCompletionChoice, StreamChoice, DeltaMessage, Usage
from .base import BaseProvider
from .pow import solve_pow_challenge


class DeepSeekProvider(BaseProvider):
    BASE_URL = "https://chat.deepseek.com"
    
    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(token=token, base_url=base_url or self.BASE_URL)
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        # /v2 透传会话：复用同一上游会话 + parent_message_id 串联多轮
        self._pt_lock = asyncio.Lock()
        self._pt_session_id: Optional[str] = None
        self._pt_parent_id: Optional[str] = None
        self._last_assistant_msg_id: Optional[str] = None
        self.headers = {
            "accept": "*/*",
            "content-type": "application/json",
            "x-client-locale": "zh_CN",
            "x-client-platform": "web",
            "x-client-version": "1.7.0",
            "x-app-version": "20241129.1",
            "x-client-timezone-offset": "28800",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "origin": "https://chat.deepseek.com",
            "referer": "https://chat.deepseek.com/",
        }
        if token:
            self.headers["authorization"] = f"Bearer {token}"
    
    @property
    def name(self) -> str:
        return "deepseek"
    
    @property
    def models(self) -> List[str]:
        return [
            "deepseek-flash",
            "deepseek-reasoner",
            "deepseek-chat",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek",
            "deepseek-r1",
            # 智能搜索变体：任意模型名加 -search 后缀即开启联网搜索
            # （如 deepseek-flash-search；与思考组合：deepseek-reasoner-search）
            "deepseek-flash-search",
            "deepseek-reasoner-search",
        ]
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def _create_session(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            f"{self.base_url}/api/v0/chat_session/create",
            headers=self.headers,
            json={}
        )
        response.raise_for_status()
        data = response.json()
        # 2026-10 观测到新版结构：data.biz_data.chat_session.id；旧版为 data.biz_data.id
        try:
            biz = data["data"]["biz_data"]
            sid = (biz.get("chat_session") or biz).get("id")
        except (KeyError, TypeError):
            sid = None
        if not sid:
            raise RuntimeError(
                "DeepSeek 会话创建返回非预期结构（多为 Token 失效或账号触发风控），"
                f"实际响应：{json.dumps(data, ensure_ascii=False)[:500]}"
            )
        return sid
    
    async def _get_pow_challenge(self, client: httpx.AsyncClient, target_path: str = "/api/v0/chat/completion") -> Dict:
        response = await client.post(
            f"{self.base_url}/api/v0/chat/create_pow_challenge",
            headers=self.headers,
            json={"target_path": target_path}
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["data"]["biz_data"]["challenge"]
        except (KeyError, TypeError):
            raise RuntimeError(
                "DeepSeek PoW 挑战获取返回非预期结构（多为 Token 失效或账号触发风控），"
                f"实际响应：{json.dumps(data, ensure_ascii=False)[:500]}"
            ) from None
    
    def _solve_pow(self, challenge: Dict) -> str:
        return solve_pow_challenge(challenge)
    
    def _build_messages(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"[System]: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(msg.content)
            elif msg.role == "assistant":
                prompt_parts.append(f"[Assistant]: {msg.content}")
        
        return "\n".join(prompt_parts)
    
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        client = await self._get_client()
        
        session_id = await self._create_session(client)
        challenge = await self._get_pow_challenge(client)
        pow_response = self._solve_pow(challenge)
        
        headers = {**self.headers, "x-ds-pow-response": pow_response}
        
        model_type = self._get_model_type(request.model)
        prompt = self._build_messages(request.messages)
        
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": model_type == "reasoner",
            "search_enabled": "search" in request.model.lower(),
            "preempt": False
        }

        async with client.stream(
            "POST",
            f"{self.base_url}/api/v0/chat/completion",
            headers=headers,
            json=payload
        ) as response:
            response.raise_for_status()

            content = ""
            channel = "content"  # 通道逻辑与流式一致：思考/搜索片段过滤
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if "v" in data:
                                v = data["v"]
                                if isinstance(v, str):
                                    if channel == "content":
                                        content += v
                                elif isinstance(v, dict):
                                    if "response" in v:
                                        resp = v["response"]
                                        if "fragments" in resp:
                                            for frag in resp["fragments"]:
                                                ftype = frag.get("type")
                                                if ftype in ("THINK", "SEARCH", "SEARCH_REF"):
                                                    channel = "reasoning"
                                                    continue
                                                if ftype == "RESPONSE":
                                                    channel = "content"
                                                frag_content = frag.get("content")
                                                if frag_content and channel == "content":
                                                    content += frag_content
                        except json.JSONDecodeError:
                            continue
        
        return ChatCompletionResponse(
            id=f"{session_id}@1",
            model=request.model,
            created=int(time.time()),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )
    
    async def chat_completion_stream(
        self, request: ChatCompletionRequest,
        session_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        track_message_id: bool = False,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        client = await self._get_client()

        if session_id is None:
            session_id = await self._create_session(client)
        challenge = await self._get_pow_challenge(client)
        pow_response = self._solve_pow(challenge)

        headers = {**self.headers, "x-ds-pow-response": pow_response}

        model_type = self._get_model_type(request.model)
        prompt = self._build_messages(request.messages)

        payload = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": model_type == "reasoner",
            # 联网搜索：模型名含 search 时开启（如 deepseek-flash-search）
            "search_enabled": "search" in request.model.lower(),
            "preempt": False
        }

        async with client.stream(
            "POST",
            f"{self.base_url}/api/v0/chat/completion",
            headers=headers,
            json=payload
        ) as response:
            if response.status_code >= 400:
                # 读出上游错误响应体，具体到哪个字段不合法（如 parent_message_id 类型）
                body = (await response.aread()).decode("utf-8", "replace")[:500]
                raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {body}")

            chunk_id = f"{session_id}@1"
            created = int(time.time())
            current_event = ""
            # 内容通道：思考正文以裸 str v 增量传输（kittors updates 解析确认），
            # 通道由 THINK/SEARCH 片段切到 reasoning（过滤），RESPONSE 片段切回 content。
            # 初始为 content，保证无片段标记的 flash 模型纯 str 流不受影响。
            channel = "content"

            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)

                            if track_message_id and current_event == "ready":
                                # 会话链：response_message_id 在 ready 事件的 data 顶层
                                #（结构参考 kittors/deepseek-web-api 的 updates 解析）
                                # 保持原始类型（int/str 均原样传递，不做转换）
                                rid = data.get("response_message_id")
                                if rid:
                                    self._last_assistant_msg_id = rid

                            if "v" in data:
                                v = data["v"]
                                if isinstance(v, str):
                                    if v == "FINISHED":
                                        continue
                                    if channel != "content":
                                        continue
                                    yield ChatCompletionChunk(
                                        id=chunk_id,
                                        model=request.model,
                                        created=created,
                                        choices=[
                                            StreamChoice(
                                                index=0,
                                                delta=DeltaMessage(content=v),
                                                finish_reason=None
                                            )
                                        ]
                                    )
                                elif isinstance(v, dict):
                                    if "response" in v:
                                        resp = v["response"]
                                        if "fragments" in resp:
                                            for frag in resp["fragments"]:
                                                ftype = frag.get("type")
                                                if ftype in ("THINK", "SEARCH", "SEARCH_REF"):
                                                    channel = "reasoning"
                                                    continue
                                                if ftype == "RESPONSE":
                                                    channel = "content"
                                                content = frag.get("content")
                                                if content and channel == "content":
                                                    yield ChatCompletionChunk(
                                                        id=chunk_id,
                                                        model=request.model,
                                                        created=created,
                                                        choices=[
                                                            StreamChoice(
                                                                index=0,
                                                                delta=DeltaMessage(content=content),
                                                                finish_reason=None
                                                            )
                                                        ]
                                                    )
                        except json.JSONDecodeError:
                            continue

        yield ChatCompletionChunk(
            id=chunk_id,
            model=request.model,
            created=created,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ]
        )

    async def pt_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """/v2 透传会话流：复用同一上游会话，parent_message_id 串联多轮，
        每轮只发增量消息（历史由上游会话维护）。单会话非并发安全，串行化。"""
        async with self._pt_lock:
            try:
                client = await self._get_client()
                if self._pt_session_id is None:
                    self._pt_session_id = await self._create_session(client)
                    print(f"[deepseek-pt] 透传会话已创建: {self._pt_session_id}")
                self._last_assistant_msg_id = None
                async for chunk in self.chat_completion_stream(
                    request,
                    session_id=self._pt_session_id,
                    parent_message_id=self._pt_parent_id,
                    track_message_id=True,
                ):
                    yield chunk
                if self._last_assistant_msg_id:
                    if self._last_assistant_msg_id != self._pt_parent_id:
                        print(f"[deepseek-pt] 已提取 assistant 消息 id: {self._last_assistant_msg_id!r} "
                              f"(type={type(self._last_assistant_msg_id).__name__})，会话链就绪")
                    self._pt_parent_id = self._last_assistant_msg_id
                else:
                    print("[deepseek-pt] 警告：本次流未提取到 assistant 消息 id，下一轮上下文可能断裂")
            except BaseException:
                # 流失败后会话状态可能已不一致（如 parent id 被上游 422 拒绝），重置自愈
                self.reset_pt()
                raise

    def reset_pt(self):
        self._pt_session_id = None
        self._pt_parent_id = None
    
    def _get_model_type(self, model: str) -> str:
        model_lower = model.lower()
        if "reasoner" in model_lower or "r1" in model_lower or "think" in model_lower:
            return "reasoner"
        if "v4-pro" in model_lower or "v4.5" in model_lower:
            return "reasoner"
        return "chat"
    
    def get_model_mapping(self, model: str) -> str:
        model_lower = model.lower()
        if "v4-pro" in model_lower:
            return "deepseek-chat"
        if "v4-flash" in model_lower:
            return "deepseek-chat"
        if "chat" in model_lower:
            return "deepseek-chat"
        return "deepseek-chat"
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
