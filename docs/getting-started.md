# Getting Started

## Installation

Install LLLM using pip:

```bash
pip install lllm
```

## Basic Usage

Here is a simple example of how to create a chat agent:

```python
from lllm import AgentBase, Prompt, register_prompt

# 1. Define a Prompt
simple_prompt = Prompt(
    path="simple_chat",
    prompt="You are a helpful assistant. User says: {user_input}"
)
register_prompt(simple_prompt)

# 2. Define an Agent
class SimpleAgent(AgentBase):
    agent_type = "simple"
    agent_group = ["assistant"]
    
    def call(self, task: str, **kwargs):
        # Initialize dialog with user input
        dialog = self.agents["assistant"].init_dialog({"user_input": task})
        # Call the agent
        response, dialog, _ = self.agents["assistant"].call(dialog)
        return response.content

# 3. Configure and Run
config = {
    "name": "simple_chat_agent",
    "log_dir": "./logs",
    "log_type": "localfile",
    "agent_configs": {
        "assistant": {
            "model_name": "gpt-4o-mini",
            "system_prompt_path": "simple_chat",
            "temperature": 0.7,
        }
    }
}

agent = SimpleAgent(config, ckpt_dir="./ckpt")
print(agent("Hello!"))
```

## Next Steps

- Learn about [Agents](core/agents.md)
- Explore [Prompts](core/prompts.md)
- Check out [Examples](https://github.com/yourusername/lllm/tree/main/examples)
