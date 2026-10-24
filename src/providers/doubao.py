import httpx
import json
import time
import uuid
import random
import string
import hashlib
import re
import asyncio
from typing import Optional, AsyncGenerator, Dict, Any, List
from dataclasses import dataclass, field

from ..models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk
from ..models import ChatMessage, ChatCompletionChoice, StreamChoice, DeltaMessage, Usage
from .base import BaseProvider


def generate_device_id() -> str:
    return ''.join(random.choices(string.digits, k=19))


def generate_web_id() -> str:
    return ''.join(random.choices(string.digits, k=19))


def generate_local_id() -> str:
    return f"local_{int(time.time() * 1000)}{random.randint(100000, 999999)}"


def generate_uuid() -> str:
    return str(uuid.uuid4())


def generate_unique_key() -> str:
    return str(uuid.uuid4())


def extract_image_urls_from_block(content: Any) -> List[str]:
    """在图片 block（block_type=2074）的 content 里递归找 {xxx: {url: ...}} 形态的字段。

    已知结构含 image_thumb / image_ori / image_ori_raw（掘金逆向文章确认），
    同一张图的多种尺寸只保留最大原图：有 raw 取 raw，其次 ori，最后 thumb；
    多张图（各自的 raw）全部保留。
    """
    found: List[tuple] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, dict) and isinstance(v.get("url"), str) and v["url"].startswith("http"):
                    found.append((str(k), v["url"]))
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(content)
    for group in ("raw", "ori", "thumb"):
        picked = [u for k, u in found if group in k]
        if picked:
            seen, out = set(), []
            for u in picked:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
            return out
    return []


# 豆包响应文本里的图片字段（掘金逆向文章确认的结构）：
# 只取无水印原图 image_ori_raw（thumb/ori 是同图小图，避免重复输出）
IMAGE_URL_RE = re.compile(r'"image_ori_raw"\s*:\s*\{\s*"url"\s*:\s*"([^"]+)"')


@dataclass
class DoubaoChatSession:
    conversation_id: str = ""
    section_id: str = ""
    local_conversation_id: str = field(default_factory=generate_local_id)


class DoubaoProvider(BaseProvider):
    BASE_URL = "https://www.doubao.com"
    
    DEFAULT_BOT_ID = "7338286299411103781"
    
    FAKE_HEADERS = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://www.doubao.com",
        "Referer": "https://www.doubao.com/chat/",
        "Sec-Ch-Ua": '"Chromium";v="144", "Google Chrome";v="144", "Not(A:Brand";v="8"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
    }
    
    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(token=token, base_url=base_url or self.BASE_URL)
        self._client: Optional[httpx.AsyncClient] = None
        self._device_id: str = generate_device_id()
        self._web_id: str = generate_web_id()
        self._tea_uuid: str = generate_web_id()
        self._fp: str = f"verify_{generate_uuid().replace('-', '_')[:20]}"
        self._sessions: Dict[str, DoubaoChatSession] = {}
        # /v2 透传会话：固定 key 复用同一豆包会话（单会话非并发安全，串行化）
        self._pt_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "doubao"
    
    @property
    def models(self) -> List[str]:
        return [
            "doubao",
            "doubao-seed-2.1-pro",
            "doubao-seed-2.1-turbo",
            "doubao-seed-2.0-pro",
            "doubao-pro",
            "doubao-lite",
            "doubao-pro-v1",
            "doubao-lite-4k",
            "doubao-lite-32k",
            "doubao-1.5-pro",
            "doubao-1.5-lite",
            "doubao-seedream-3",
        ]
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client
    
    def _get_cookies_dict(self) -> Dict[str, str]:
        cookies = {}
        if self.token:
            cookies["sessionid"] = self.token
            cookies["sessionid_ss"] = self.token
        return cookies
    
    def _get_or_create_session(self, session_key: str) -> DoubaoChatSession:
        if session_key not in self._sessions:
            self._sessions[session_key] = DoubaoChatSession()
        return self._sessions[session_key]
    
    def _build_query_params(self) -> Dict[str, str]:
        return {
            "aid": "497858",
            "device_id": self._device_id,
            "device_platform": "web",
            "fp": self._fp,
            "language": "zh",
            "pc_version": "3.5.10",
            "pkg_type": "release_version",
            "real_aid": "497858",
            "region": "",
            "samantha_web": "1",
            "sys_region": "",
            "tea_uuid": self._tea_uuid,
            "use-olympus-account": "1",
            "version_code": "20800",
            "web_id": self._web_id,
            "web_tab_id": str(uuid.uuid4()),
        }
    
    def _build_request_body(
        self, 
        messages: List[ChatMessage], 
        session: DoubaoChatSession,
        model: str = "doubao"
    ) -> Dict[str, Any]:
        local_message_id = generate_uuid()
        block_id = generate_uuid()
        
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                last_user_message = msg.content
                break
        
        if not last_user_message:
            last_user_message = messages[-1].content if messages else ""
        
        request_body = {
            "client_meta": {
                "local_conversation_id": session.local_conversation_id,
                "conversation_id": session.conversation_id,
                "bot_id": self.DEFAULT_BOT_ID,
                "last_section_id": session.section_id,
                "last_message_index": None
            },
            "messages": [
                {
                    "local_message_id": local_message_id,
                    "content_block": [
                        {
                            "block_type": 10000,
                            "content": {
                                "text_block": {
                                    "text": last_user_message,
                                    "icon_url": "",
                                    "icon_url_dark": "",
                                    "summary": ""
                                },
                                "pc_event_block": ""
                            },
                            "block_id": block_id,
                            "parent_id": "",
                            "meta_info": [],
                            "append_fields": []
                        }
                    ],
                    "message_status": 0
                }
            ],
            "option": {
                "send_message_scene": "",
                "create_time_ms": int(time.time() * 1000),
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": False,
                "tts_switch": False,
                "need_deep_think": 0,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": generate_unique_key(),
                "start_seq": 0,
                "need_create_conversation": not bool(session.conversation_id),
                "conversation_init_option": {"need_ack_conversation": True},
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False
            },
            "ext": {
                "use_deep_think": "0",
                "fp": self._fp,
                "conversation_init_option": '{"need_ack_conversation":true}',
                "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1"
            }
        }
        
        return request_body
    
    def _parse_sse_response(self, content: str) -> List[Dict[str, Any]]:
        events = []
        lines = content.split("\n")
        current_event = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_event:
                    events.append(current_event)
                    current_event = {}
                continue
            
            if line.startswith("id:"):
                current_event["id"] = line[3:].strip()
            elif line.startswith("event:"):
                current_event["event"] = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    current_event["data"] = json.loads(data_str)
                except json.JSONDecodeError:
                    current_event["data"] = data_str
        
        if current_event:
            events.append(current_event)
        
        return events
    
    def _extract_content_from_events(self, events: List[Dict], session: DoubaoChatSession) -> str:
        content = ""
        
        for event in events:
            event_type = event.get("event", "")
            data = event.get("data", {})
            
            if event_type == "SSE_ACK":
                if isinstance(data, dict):
                    ack_meta = data.get("ack_client_meta", {})
                    if not session.conversation_id:
                        session.conversation_id = ack_meta.get("conversation_id", "")
                    if not session.section_id:
                        session.section_id = ack_meta.get("section_id", "")
            
            elif event_type == "CHUNK_DELTA":
                if isinstance(data, dict):
                    text = data.get("text", "")
                    content += text
            
            elif event_type == "STREAM_MSG_NOTIFY":
                if isinstance(data, dict):
                    meta = data.get("meta", {})
                    msg_data = data.get("content", {})
                    content_blocks = msg_data.get("content_block", [])
                    for block in content_blocks:
                        if block.get("block_type") == 10000:
                            text_block = block.get("content", {}).get("text_block", {})
                            text = text_block.get("text", "")
                            if text and not content:
                                content = text
        
        return content
    
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        client = await self._get_client()
        
        session_key = f"{request.model}_{hash(tuple(m.content for m in request.messages))}"
        session = self._get_or_create_session(session_key)
        
        query_params = self._build_query_params()
        request_body = self._build_request_body(request.messages, session, request.model)
        
        headers = {
            **self.FAKE_HEADERS,
            "Content-Type": "application/json",
        }
        
        response = await client.post(
            f"{self.base_url}/chat/completion",
            params=query_params,
            headers=headers,
            json=request_body,
            cookies=self._get_cookies_dict()
        )
        response.raise_for_status()
        
        raw_content = response.text
        events = self._parse_sse_response(raw_content)
        content = self._extract_content_from_events(events, session)
        
        return ChatCompletionResponse(
            id=session.conversation_id or f"chatcmpl-{uuid.uuid4().hex[:8]}",
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
        session_key: Optional[str] = None,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        client = await self._get_client()

        if session_key is None:
            session_key = f"{request.model}_{hash(tuple(m.content for m in request.messages))}"
        session = self._get_or_create_session(session_key)
        
        query_params = self._build_query_params()
        request_body = self._build_request_body(request.messages, session, request.model)
        
        headers = {
            **self.FAKE_HEADERS,
            "Content-Type": "application/json",
        }
        
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completion",
            params=query_params,
            headers=headers,
            json=request_body,
            cookies=self._get_cookies_dict()
        ) as response:
            response.raise_for_status()

            buffer = ""
            full_parts: List[str] = []   # 原始响应文本（流结束时兜底扫描图片 URL）
            emitted_urls: set = set()    # 已输出的图片 URL（block 路径与兜底扫描去重）
            # 空流诊断：豆包解析器只认 4 种事件，上游改结构/返回错误时会被静默忽略，
            # 记录各事件类型的首个载荷样本，零内容时回传给客户端定位
            saw_content = False
            type_samples: Dict[str, str] = {}
            seen_block_types: set = set()
            seen_delta_keys: set = set()
            unknown_event_samples: Dict[str, str] = {}   # 未处理事件样本（回传对话定位）
            delta_key_samples: Dict[tuple, str] = {}     # 非 text 键组合的 delta 样本
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            raw_head = ""  # 响应体开头（零事件时用于判断返回的是 JSON/HTML/空）

            def _note_block(block: Dict[str, Any]) -> None:
                bt = block.get("block_type")
                if bt not in (10000, 2074) and bt not in seen_block_types:
                    seen_block_types.add(bt)
                    try:
                        sample = json.dumps(block, ensure_ascii=False)[:300]
                    except (TypeError, ValueError):
                        sample = str(block)[:300]
                    print(f"[doubao] 未处理的 block_type={bt}，样本: {sample}")

            def _text_from_block(block: Dict[str, Any]) -> str:
                if block.get("block_type") == 2074:
                    urls = extract_image_urls_from_block(block.get("content", {}))
                    emitted_urls.update(urls[:3])
                    return "\n".join(f"![图片]({u})" for u in urls[:3])
                tb = block.get("content", {}).get("text_block", {})
                return tb.get("text", "")

            # 块状态机：block_id -> 已输出文本长度。同一块经 STREAM_MSG_NOTIFY /
            # STREAM_CHUNK patch 多次到达时只输出新增部分，避免快照重复
            block_progress: Dict[str, int] = {}

            def _emit_blocks(content_blocks: List[Dict[str, Any]]) -> List[str]:
                segs = []
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    _note_block(block)
                    bid = str(block.get("block_id", id(block)))
                    text = _text_from_block(block)
                    if not text:
                        continue
                    prev_len = block_progress.get(bid, 0)
                    if len(text) > prev_len:
                        segs.append(text[prev_len:])
                        block_progress[bid] = len(text)
                    # 变短视为替换/重置语义，保守忽略防重复
                return segs

            def _flush_images() -> List[str]:
                """流结束时从原始响应文本兜底扫描图片 URL（block 结构未命中时生效）。"""
                new_urls = []
                for m in IMAGE_URL_RE.finditer("".join(full_parts)):
                    u = m.group(1).replace("\\/", "/")
                    if u not in emitted_urls:
                        emitted_urls.add(u)
                        new_urls.append(u)
                if new_urls:
                    print(f"[doubao] block 结构未命中图片，已从原始流文本兜底提取 {len(new_urls)} 个 URL"
                          f"（首个: {new_urls[0][:120]}）")
                return new_urls

            def _sample(event_type: str, payload: Any) -> None:
                if event_type in type_samples:
                    return
                try:
                    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
                except (TypeError, ValueError):
                    text = str(payload)
                type_samples[event_type or "<无事件名>"] = text[:160]
                # 未处理的事件类型打日志并留存样本（可能携带正文内容），流结束时回传对话便于定位。
                # 已知无需处理：SSE_HEARTBEAT=心跳，FULL_MSG_NOTIFY=完整消息快照（与增量重复）
                if event_type and event_type not in (
                    "SSE_ACK", "STREAM_MSG_NOTIFY", "CHUNK_DELTA", "SSE_REPLY_END",
                    "STREAM_CHUNK", "SSE_HEARTBEAT", "FULL_MSG_NOTIFY",
                ):
                    unknown_event_samples[event_type] = text[:200]
                    print(f"[doubao] 未处理事件 {event_type}，样本: {text[:400]}")

            def _diag_chunk() -> ChatCompletionChunk:
                samples = " ｜ ".join(
                    f"{t}: {s}" for t, s in list(type_samples.items())[:5]
                )
                meta = f"HTTP {status_code}，Content-Type: {content_type or '<无>'}，响应开头：{raw_head[:300] or '<空>'}"
                if samples:
                    detail = f"事件样本：{samples}；响应：{meta}"
                else:
                    detail = f"完全没有 SSE 事件。响应：{meta}"
                return ChatCompletionChunk(
                    id=chunk_id,
                    model=request.model,
                    created=created,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content=(
                                "[豆包调试] 流结束但未解析到任何内容（上游可能改了响应结构或返回了错误）。"
                                + detail
                            )),
                            finish_reason="stop"
                        )
                    ]
                )

            def _struct_diag_text() -> str:
                """未处理结构样本（引用词/分段信息可能藏在这些结构里）。"""
                parts = []
                if unknown_event_samples:
                    parts.append("未处理事件：" + " ｜ ".join(
                        f"{k}: {v[:130]}" for k, v in list(unknown_event_samples.items())[:3]))
                if delta_key_samples:
                    parts.append("delta键变体：" + " ｜ ".join(
                        v[:130] for v in list(delta_key_samples.values())[:3]))
                return " ｜ ".join(parts)

            async for chunk in response.aiter_text():
                full_parts.append(chunk)
                if len(raw_head) < 300:
                    raw_head = (raw_head + chunk)[:300]
                buffer += chunk

                while "\n\n" in buffer:
                    event_str, buffer = buffer.split("\n\n", 1)

                    lines = event_str.strip().split("\n")
                    event_type = ""
                    data = {}

                    for line in lines:
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                data = {}
                                _sample(f"{event_type or '?'}(非JSON)", data_str)

                    _sample(event_type, data)

                    if event_type == "SSE_ACK":
                        if isinstance(data, dict):
                            ack_meta = data.get("ack_client_meta", {})
                            if not session.conversation_id:
                                session.conversation_id = ack_meta.get("conversation_id", "")
                                chunk_id = session.conversation_id
                            if not session.section_id:
                                session.section_id = ack_meta.get("section_id", "")

                    elif event_type == "STREAM_MSG_NOTIFY":
                        if isinstance(data, dict):
                            content = data.get("content", {})
                            for seg in _emit_blocks(content.get("content_block", [])):
                                saw_content = True
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    model=request.model,
                                    created=created,
                                    choices=[
                                        StreamChoice(
                                            index=0,
                                            delta=DeltaMessage(content=seg),
                                            finish_reason=None
                                        )
                                    ]
                                )

                    elif event_type == "STREAM_CHUNK":
                        # patch 操作数组：patch_value.content_block 携带块更新
                        #（带引用词的正文块经此下发），与 NOTIFY 共用块状态机
                        if isinstance(data, dict):
                            for op in data.get("patch_op", []) or []:
                                if not isinstance(op, dict):
                                    continue
                                pv = op.get("patch_value")
                                if isinstance(pv, dict):
                                    for seg in _emit_blocks(pv.get("content_block", [])):
                                        saw_content = True
                                        yield ChatCompletionChunk(
                                            id=chunk_id,
                                            model=request.model,
                                            created=created,
                                            choices=[
                                                StreamChoice(
                                                    index=0,
                                                    delta=DeltaMessage(content=seg),
                                                    finish_reason=None
                                                )
                                            ]
                                        )

                    elif event_type == "CHUNK_DELTA":
                        if isinstance(data, dict):
                            keys = tuple(sorted(data.keys()))
                            if keys not in seen_delta_keys:
                                seen_delta_keys.add(keys)
                                try:
                                    sample = json.dumps(data, ensure_ascii=False)[:400]
                                except (TypeError, ValueError):
                                    sample = str(data)[:400]
                                print(f"[doubao] CHUNK_DELTA 键组合 {keys}，样本: {sample}")
                                if keys != ("text",):
                                    delta_key_samples[keys] = sample[:200]
                            text = data.get("text", "")
                            if text:
                                saw_content = True
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    model=request.model,
                                    created=created,
                                    choices=[
                                        StreamChoice(
                                            index=0,
                                            delta=DeltaMessage(content=text),
                                            finish_reason=None
                                        )
                                    ]
                                )

                    elif event_type == "SSE_REPLY_END":
                        extra = _flush_images()
                        if extra:
                            saw_content = True
                            yield ChatCompletionChunk(
                                id=chunk_id,
                                model=request.model,
                                created=created,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=DeltaMessage(content="\n" + "\n".join(f"![图片]({u})" for u in extra[:5])),
                                        finish_reason=None
                                    )
                                ]
                            )
                        if not saw_content:
                            yield _diag_chunk()
                        sd = _struct_diag_text()
                        if sd:
                            yield ChatCompletionChunk(
                                id=chunk_id,
                                model=request.model,
                                created=created,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta=DeltaMessage(content="\n\n---\n[豆包调试] 检测到未解析结构（可能含引用词/分段信息）：" + sd),
                                        finish_reason=None
                                    )
                                ]
                            )
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
                        return

            extra = _flush_images()
            if extra:
                saw_content = True
                yield ChatCompletionChunk(
                    id=chunk_id,
                    model=request.model,
                    created=created,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content="\n" + "\n".join(f"![图片]({u})" for u in extra[:5])),
                            finish_reason=None
                        )
                    ]
                )
            sd = _struct_diag_text()
            if sd:
                saw_content = True
                yield ChatCompletionChunk(
                    id=chunk_id,
                    model=request.model,
                    created=created,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta=DeltaMessage(content="\n\n---\n[豆包调试] 检测到未解析结构（可能含引用词/分段信息）：" + sd),
                            finish_reason=None
                        )
                    ]
                )
        if not saw_content:
            yield _diag_chunk()
    
    def get_model_mapping(self, model: str) -> str:
        model_lower = model.lower()
        if "seed-2.1" in model_lower or "seed-2-1" in model_lower:
            if "turbo" in model_lower:
                return "Doubao-Seed-2.1-turbo"
            return "Doubao-Seed-2.1-pro"
        if "seed-2.0" in model_lower or "seed-2-0" in model_lower:
            return "Doubao-Seed-2-0-Pro"
        if "seedream" in model_lower:
            return "Doubao-Seed-2-0-Pro"
        if "pro-v1" in model_lower or "1.5-pro" in model_lower:
            return "Doubao-pro"
        if "lite-4k" in model_lower or "lite-32k" in model_lower:
            return "Doubao-lite"
        if "lite" in model_lower:
            return "Doubao-lite"
        if "pro" in model_lower:
            return "Doubao-pro"
        return "doubao"
    
    async def pt_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionChunk, None]:
        """/v2 透传会话流：固定复用同一豆包会话（conversation_id 由 SSE_ACK 提取后续期）。"""
        async with self._pt_lock:
            async for chunk in self.chat_completion_stream(request, session_key="passthrough"):
                yield chunk

    def reset_pt(self):
        self._sessions.pop("passthrough", None)

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
