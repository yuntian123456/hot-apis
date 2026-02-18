from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List, Dict, Any
from ..models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk


class BaseProvider(ABC):
    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        self.token = token
        self.base_url = base_url
        self.headers: Dict[str, str] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def models(self) -> List[str]:
        pass
    
    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        pass
    
    def get_model_mapping(self, model: str) -> str:
        return model
    
    def supports_model(self, model: str) -> bool:
        return model in self.models or any(
            model.startswith(m.split("-")[0]) for m in self.models
        )
