from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
import datetime as dt
import tiktoken
from tiktoken.model import encoding_name_for_model


class RCollections(Enum):
    DIALOGS = 'dialogs' # To track the dialogs created in a session, and context for each llm call
    FRONTEND = 'frontend' # To track the frontend info in between for replay in Streamlit
    MESSAGES = 'messages' # To track the messages created in a session


class ParseError(Exception):
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.message)

class Roles(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    TOOL = 'tool'

    @property
    def openai(self):
        # https://cdn.openai.com/spec/model-spec-2024-05-08.html#definitions
        if self == Roles.SYSTEM:
            return 'developer'
        elif self == Roles.ASSISTANT:
            return 'assistant'
        elif self == Roles.USER:
            return 'user'
        elif self == Roles.TOOL:
            return 'tool'
        
    @property
    def anthropic(self):
        raise NotImplementedError("Anthropic roles not implemented")
    
    @property
    def gemini(self):
        raise NotImplementedError("Gemini roles not implemented")


class Providers(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GEMINI = 'gemini'

class Modalities(Enum):
    TEXT = 'text'
    IMAGE = 'image'
    AUDIO = 'audio' 
    FUNCTION_CALL = 'function_call'
    # pdf?
    
class Roles(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'
    TOOL_CALL = 'tool_call'

LLM_SIDE_ROLES = [Roles.ASSISTANT, Roles.TOOL_CALL]
    

class Features(Enum):
    FUNCTION_CALL = 'function_call'
    STRUCTURED_OUTPUT = 'structured_output'
    STREAMING = 'streaming'
    FINETUNING = 'finetuning'
    DISTILLATION = 'distillation'
    PREDICTED_OUTPUT = 'predicted_output'
    CLASSIFICATION = 'classification'


MODEL_CARDS = {}



class APITypes(Enum):
    COMPLETION = 'completion'
    RESPONSE = 'response'  


@dataclass
class Snapshot:
    name: str
    date: str 

    @property
    def dt(self):
        return dt.datetime.strptime(self.date, '%Y-%m-%d')


# See more in https://platform.openai.com/docs/api-reference/chat/create
OPENAI_ARGS = [
    'temperature', 
    'max_completion_tokens',
    'presence_penalty',
    'reasoning_effort', # o series only
    'response_format', # for structured output
    'tools', # for function calling
    'tool_choice', # for function calling
    'logit_bias', # for classification
    'top_logprobs', # for classification
]


OPENAI_ENCODINGS = {
    'gpt-4.1': 'o200k_base',
    'o4-mini': 'o200k_base',
    'gpt-4.1-mini': 'o200k_base',
}

@dataclass
class CompletionCost:
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    cost: float

    def __str__(self):
        return f'''
Prompt tokens: {self.prompt_tokens}, 
Completion tokens: {self.completion_tokens}, 
Cached prompt tokens: {self.cached_prompt_tokens}, 
Cost: {self.cost:.4f} USD
        '''



@dataclass
class ModelCard:
    name: str
    provider: Providers
    snapshots: List[Snapshot]
    max_tokens: int
    max_output_tokens: int
    input_price: float # per 1M tokens
    cached_input_price: float # per 1M tokens
    output_price: float # per 1M tokens
    knowledge_cutoff: str = None # YYYY-MM-DD, one day after
    features: List[Features] = field(default_factory=lambda: [])
    input_modalities: List[Modalities] = field(default_factory=lambda: [Modalities.TEXT])
    is_reasoning: bool = False
    base_url: str = None

    @property
    def snapshot_dict(self):
        return {s.name: s for s in self.snapshots}

    def __post_init__(self):
        self.snapshots = sorted(self.snapshots, key=lambda x: x.dt)
        MODEL_CARDS[self.name] = self

    @property
    def latest_snapshot(self):
        return self.snapshots[-1]

    def check_args(self, args: Dict[str, Any]):
        if self.provider == Providers.OPENAI:
            supported_args = OPENAI_ARGS
        elif self.provider == Providers.ANTHROPIC:
            raise NotImplementedError("Anthropic args not supported")
        elif self.provider == Providers.GEMINI:
            raise NotImplementedError("Gemini args not supported")

        for arg in args:
            if arg not in supported_args:
                raise ValueError(f"Argument {arg} not supported")

    def cost(self, usage: Dict[str, float]) -> CompletionCost:
        if self.provider == Providers.OPENAI:
            return openai_model_usage(self, usage)
        elif self.provider == Providers.ANTHROPIC:
            return anthropic_model_usage(self, usage)
        elif self.provider == Providers.GEMINI:
            return gemini_model_usage(self, usage)
        
    def tokenize(self, text: str) -> List[int]:
        if self.provider == Providers.OPENAI:
            return tokenize_openai(text, self.name)
        elif self.provider == Providers.ANTHROPIC:
            return tokenize_anthropic(text, self.name)
        elif self.provider == Providers.GEMINI:
            return tokenize_gemini(text, self.name)
        
    def make_classifier(self, classes: List[str], strength: int = 10) -> Dict[str, Any]:
        assert Features.CLASSIFICATION in self.features, f'Model {self.name} does not support classification'
        token_ids = self.tokenize(' '.join(classes))
        assert len(token_ids) == len(classes), f'Classes {classes} cannot be tokenized into single tokens'
        logit_bias = {i: strength for i in token_ids}
        if self.provider == Providers.OPENAI:
            return classifier_args_openai(logit_bias)
        elif self.provider == Providers.ANTHROPIC:
            raise NotImplementedError("Anthropic classifier args not supported")
        elif self.provider == Providers.GEMINI:
            raise NotImplementedError("Gemini classifier args not supported")




def tokenize_openai(text: str, encoding: str = 'o200k_base'):
    # Models to encoding: https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
    # List of all the encoding names: tiktoken.registry.list_encoding_names()   
    if encoding in OPENAI_ENCODINGS:
        encoding = OPENAI_ENCODINGS[encoding]
    elif encoding not in tiktoken.registry.list_encoding_names():
        try:
            encoding = encoding_name_for_model(encoding)
        except:
            raise ValueError(f"Encoding {encoding} not found")
    enc = tiktoken.get_encoding(encoding)
    return enc.encode(text)

def tokenize_anthropic(text: str, encoding: str):
    raise NotImplementedError("anthropic_tokenize not implemented")

def tokenize_gemini(text: str, encoding: str):
    raise NotImplementedError("gemini_tokenize not implemented")

def classifier_args_openai(logit_bias: Dict[int, int]):
    return {
        'logit_bias': logit_bias,
        'temperature': 0.0, # temperature=0.0, 
        'top_logprobs': len(logit_bias),
        'max_completion_tokens': 3, # give some buffer
        'logprobs': True,
    }

            
def find_model_card(name: str) -> ModelCard:
    if name not in MODEL_CARDS:
        for model_card in MODEL_CARDS.values():
            if name in model_card.snapshot_dict:
                return model_card
        raise ValueError(f"Model card {name} not found")
    return MODEL_CARDS[name]


def openai_model_usage(model_card: ModelCard, usage: Dict[str, float]) -> CompletionCost:
    # {
    #   "completion_tokens":400,
    #   "prompt_tokens":13584,
    #   "total_tokens":13984,
    #   "completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,"reasoning_tokens":0,"rejected_prediction_tokens":0},
    #   "prompt_tokens_details":{"audio_tokens":0,"cached_tokens":0}
    # }
    cost = 0
    if usage['prompt_tokens_details'] is None:
        cached_tokens = 0
    else:
        cached_tokens = usage['prompt_tokens_details']['cached_tokens']
    cost += model_card.input_price * (usage['prompt_tokens']-cached_tokens)
    cost += model_card.output_price * usage['completion_tokens']
    cost += model_card.cached_input_price * cached_tokens
    cost = cost / 1000000 # convert to USD
    return CompletionCost(
        prompt_tokens=usage['prompt_tokens'],
        completion_tokens=usage['completion_tokens'],
        cached_prompt_tokens=cached_tokens,
        cost=cost
    )

def anthropic_model_usage(model_card: ModelCard, usage: Dict[str, float]) -> CompletionCost:
    raise NotImplementedError("anthropic_usage not implemented")

def gemini_model_usage(model_card: ModelCard, usage: Dict[str, float]) -> CompletionCost:
    raise NotImplementedError("gemini_usage not implemented")



GPT_41 = ModelCard(
    name='gpt-4.1',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='gpt-4.1-2025-04-14', date='2025-04-14'),
    ],
    max_tokens=1047576,
    max_output_tokens=32768,
    input_price=2,
    cached_input_price=0.5,
    output_price=8,
    knowledge_cutoff='2024-06-01',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features = [
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
        Features.FINETUNING, 
        Features.DISTILLATION, 
        Features.PREDICTED_OUTPUT,
        Features.CLASSIFICATION,
    ],
)
    
O4_MINI = ModelCard(
    name='o4-mini',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='o4-mini-2025-04-16', date='2025-04-16'),
    ],
    max_tokens=200000,
    max_output_tokens=100000,
    input_price=1.1,
    cached_input_price=0.275,
    output_price=4.4,
    knowledge_cutoff='2024-06-01',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
    ],
    is_reasoning=True,
)

O3 = ModelCard(
    name='o3',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='o3-2025-04-16', date='2025-04-16'),
    ],
    max_tokens=200000,
    max_output_tokens=100000,
    input_price=2,
    cached_input_price=0.5,
    output_price=8,
    knowledge_cutoff='2024-06-01',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
    ],
    is_reasoning=True,
)

GPT_41_MINI = ModelCard(
    name='gpt-4.1-mini',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='gpt-4.1-mini-2025-04-14', date='2025-04-14'),
    ],
    max_tokens=1047576,
    max_output_tokens=32768,
    input_price=0.4,
    cached_input_price=0.1,
    output_price=1.6,
    knowledge_cutoff='2024-06-01',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
        Features.FINETUNING, 
        Features.CLASSIFICATION,
    ]
)



GPT_4O_MINI = ModelCard(
    name='gpt-4o-mini',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='gpt-4o-mini-2024-07-18', date='2024-07-18'),
    ],
    max_tokens=128000,
    max_output_tokens=16384,
    input_price=0.15,
    cached_input_price=0.075,
    output_price=0.6,
    knowledge_cutoff='2023-10-01',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
        Features.FINETUNING, 
        Features.CLASSIFICATION,
    ]
)



GPT_5_MINI = ModelCard(
    name='gpt-5-mini',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='gpt-5-mini-2025-08-07', date='2025-08-07'),
    ],
    max_tokens=400000,
    max_output_tokens=128000,
    input_price=0.25,
    cached_input_price=0.025,
    output_price=2,
    knowledge_cutoff='2024-05-31',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
        Features.FINETUNING, 
        Features.CLASSIFICATION,
    ]
)


GPT_51 = ModelCard(
    name='gpt-5.1',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name='gpt-5.1-2025-11-13', date='2025-11-13'),
    ],
    max_tokens=400000,
    max_output_tokens=128000,
    input_price=1.25,
    cached_input_price=0.125,
    output_price=10,
    knowledge_cutoff='2024-09-30',
    input_modalities=[Modalities.TEXT, Modalities.IMAGE],
    features=[
        Features.FUNCTION_CALL, 
        Features.STRUCTURED_OUTPUT, 
        Features.STREAMING,     
        Features.FINETUNING, 
        Features.CLASSIFICATION,
    ]
)




# OPENSOURCE MODELS

QWEN3_NEXT_80B_THINKING = ModelCard(
    name='qwen3-next-80b-thinking',
    provider=Providers.OPENAI, # use openai client to call this model
    snapshots=[
        Snapshot(name="Qwen/Qwen3-Next-80B-A3B-Thinking", date='2025-09-09'),
    ],
    max_tokens=262000,
    max_output_tokens=100000,
    input_price=0.15,
    cached_input_price=0.15, # together seems to price the same for cached and uncached input tokens
    output_price=1.5,
    base_url="https://api.together.xyz/v1", 
)

KIMI_K2_THINKING = ModelCard(
    name='kimi-k2-thinking',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name="moonshotai/Kimi-K2-Thinking", date='2025-11-04'),
    ],
    max_tokens=262000,
    max_output_tokens=100000,
    input_price=1.2,
    cached_input_price=1.2,
    output_price=4,
    base_url="https://api.together.xyz/v1", 
)

DEEPSEEK_V31 = ModelCard(
    name='deepseek-v3.1',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name="deepseek-ai/DeepSeek-V3.1", date='2025-09-05'),
    ],
    max_tokens=131000,
    max_output_tokens=100000,
    input_price=0.6,
    cached_input_price=0.6,
    output_price=1.7,
    base_url="https://api.together.xyz/v1", 
)

GLM_4_6 = ModelCard(
    name='glm-4.6',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name="zai-org/GLM-4.6", date='2025-09-29'),
    ],
    max_tokens=202000,
    max_output_tokens=100000,
    input_price=0.6,
    cached_input_price=0.6,
    output_price=2.2,
    base_url="https://api.together.xyz/v1", 
)

OSS_20B = ModelCard(
    name='oss-20b',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name="openai/gpt-oss-20b",date='2025-08-04'),
    ],
    max_tokens=131000,
    max_output_tokens=100000,
    input_price=0.05,
    cached_input_price=0.05,
    output_price=0.2,
    base_url="https://api.together.xyz/v1", 
)

OSS_120B = ModelCard(
    name='oss-120b',
    provider=Providers.OPENAI,
    snapshots=[
        Snapshot(name="openai/gpt-oss-120b", date='2025-08-04'),
    ],
    max_tokens=131000,
    max_output_tokens=100000,
    input_price=0.15,
    cached_input_price=0.15,
    output_price=0.6,
    base_url="https://api.together.xyz/v1", 
)