from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional

from lllm.core.models import Message, Prompt

class BaseProvider(ABC):
    @abstractmethod
    def call(
        self,
        dialog: Any,  # Typed as Any to avoid circular import with Dialog, or use 'Dialog' forward ref
        prompt: Prompt,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        extra: Optional[Dict[str, Any]] = None,
    ) -> Message:
        pass

    @abstractmethod
    def stream(
        self,
        dialog: Any,
        prompt: Prompt,
        model: str,
        model_args: Optional[Dict[str, Any]] = None,
        parser_args: Optional[Dict[str, Any]] = None,
        responder: str = 'assistant',
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, Message]:
        pass
