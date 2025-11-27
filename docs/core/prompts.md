# Prompts

In LLLM, a `Prompt` is more than just a text string. It is a structured object that defines:

- The template string (with variables like `{user_input}`).
- Available functions/tools.
- Exception handling logic.
- Interrupt handling logic (for tool calls).
- Output parsing and formatting.

## Defining a Prompt

```python
from lllm import Prompt, Function

# Define a tool
def get_weather(location: str):
    return "Sunny"

weather_tool = Function(name="get_weather", ...)
weather_tool.link_function(get_weather)

# Define the prompt
my_prompt = Prompt(
    path="weather_bot",
    prompt="You are a weather bot. User asks: {question}",
    functions_list=[weather_tool],
    exception_prompt="I encountered an error: {error_message}. Let me try again.",
)
```

## Neuro-Symbolic Features

Prompts support advanced features like recursive exception handling and structured output enforcement, enabling robust neuro-symbolic workflows.
