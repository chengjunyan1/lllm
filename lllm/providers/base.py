from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Union
from lllm.core.models import Message, Prompt
from lllm.core.const import ModelCard

class BaseProvider(ABC):
    @abstractmethod
    def call(self, 
             dialog: Any, # Typed as Any to avoid circular import with Dialog, or use 'Dialog' forward ref
             prompt: Prompt,
             model: str, 
             model_args: Dict[str, Any] = {}, 
             parser_args: Dict[str, Any] = {},
             responder: str = 'assistant', 
             extra: Dict[str, Any] = {}) -> Message:
        pass

    @abstractmethod
    def stream(self, 
               dialog: Any, 
               prompt: Prompt,
               model: str, 
               model_args: Dict[str, Any] = {}, 
               parser_args: Dict[str, Any] = {},
               responder: str = 'assistant', 
               extra: Dict[str, Any] = {}) -> Generator[Any, None, Message]:
        pass
