from enum import Enum
from typing import List, Dict, Any, Optional
import datetime as dt
import tiktoken
from tiktoken.model import encoding_name_for_model
from pydantic import BaseModel, Field, field_validator

class RCollections(str, Enum):
    DIALOGS = 'dialogs'
    FRONTEND = 'frontend'
    MESSAGES = 'messages'

class ParseError(Exception):
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.message)

class Roles(str, Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    TOOL = 'tool'
    TOOL_CALL = 'tool_call'

    @property
    def openai(self):
        if self == Roles.SYSTEM:
            return 'developer'
        return self.value

class Providers(str, Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'

class Modalities(str, Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio'
    FUNCTION_CALL = 'function_call'

class Features(str, Enum):
    FUNCTION_CALL = 'function_call'
    STRUCTURED_OUTPUT = 'structured_output'
    STREAMING = 'streaming'
    FINETUNING = 'finetuning'
    DISTILLATION = 'distillation'
    PREDICTED_OUTPUT = 'predicted_output'
    CLASSIFICATION = 'classification'
    COMPUTER_USE = 'computer_use'
    WEB_SEARCH = 'web_search'

class APITypes(str, Enum):
    COMPLETION = 'completion'
    RESPONSE = 'response'

class Snapshot(BaseModel):
    name: str
    date: str

    @property
    def dt(self):
        return dt.datetime.strptime(self.date, '%Y-%m-%d')

class CompletionCost(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    cost: float = 0.0

    def __str__(self):
        return (f"Prompt tokens: {self.prompt_tokens}, \n"
                f"Completion tokens: {self.completion_tokens}, \n"
                f"Cached prompt tokens: {self.cached_prompt_tokens}, \n"
                f"Cost: {self.cost:.4f} USD")

class ModelCard(BaseModel):
    name: str
    provider: Providers
    snapshots: List[Snapshot]
    max_tokens: int
    max_output_tokens: int
    input_price: float # per 1M tokens
    cached_input_price: float # per 1M tokens
    output_price: float # per 1M tokens
    knowledge_cutoff: Optional[str] = None
    features: List[Features] = Field(default_factory=list)
    input_modalities: List[Modalities] = Field(default_factory=lambda: [Modalities.TEXT])
    is_reasoning: bool = False
    base_url: Optional[str] = None

    @property
    def snapshot_dict(self):
        return {s.name: s for s in self.snapshots}

    @property
    def latest_snapshot(self):
        return sorted(self.snapshots, key=lambda x: x.dt)[-1]

    def check_args(self, args: Dict[str, Any]):
        # This logic might be better placed in the Provider implementation
        pass

    def cost(self, usage: Dict[str, float]) -> CompletionCost:
        # This logic should ideally be moved to a utility or provider method to avoid circular imports if utils depends on const
        # For now, keeping a simplified version or moving the logic out.
        # I will move the cost calculation logic to a separate utility to keep this file pure data/constants.
        pass

MODEL_CARDS: Dict[str, ModelCard] = {}

def register_model_card(card: ModelCard):
    MODEL_CARDS[card.name] = card

# Define standard models
GPT_41 = ModelCard(
    name='gpt-4.1',
    provider=Providers.OPENAI,
    snapshots=[Snapshot(name='gpt-4.1-2025-04-14', date='2025-04-14')],
    max_tokens=1047576,
    max_output_tokens=32768,
    input_price=2,
    cached_input_price=0.5,
    output_price=8,
    knowledge_cutoff='2024-06-01',
    features=[
        Features.FUNCTION_CALL, Features.STRUCTURED_OUTPUT, Features.STREAMING,
        Features.FINETUNING, Features.DISTILLATION, Features.PREDICTED_OUTPUT,
        Features.CLASSIFICATION
    ]
)
register_model_card(GPT_41)

# ... (I will add other models later or in a separate file to keep this clean, but for now I'll add a few key ones)

GPT_4O_MINI = ModelCard(
    name='gpt-4o-mini',
    provider=Providers.OPENAI,
    snapshots=[Snapshot(name='gpt-4o-mini-2024-07-18', date='2024-07-18')],
    max_tokens=128000,
    max_output_tokens=16384,
    input_price=0.15,
    cached_input_price=0.075,
    output_price=0.6,
    knowledge_cutoff='2023-10-01',
    features=[
        Features.FUNCTION_CALL, Features.STRUCTURED_OUTPUT, Features.STREAMING,
        Features.FINETUNING, Features.CLASSIFICATION
    ]
)
register_model_card(GPT_4O_MINI)

LLM_SIDE_ROLES = [Roles.ASSISTANT, Roles.TOOL_CALL]

def find_model_card(name: str) -> ModelCard:
    if name in MODEL_CARDS:
        return MODEL_CARDS[name]
    # Fallback or search logic could go here
    # For now, return a default or raise error
    # If name matches a snapshot, find the parent card
    for card in MODEL_CARDS.values():
        if name in card.snapshot_dict:
            return card
    
    # If not found, maybe create a generic one or raise
    # For robustness, let's return a generic card if not found, or raise
    raise ValueError(f"Model card for '{name}' not found")

