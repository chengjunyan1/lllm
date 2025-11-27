# Low-Level Language Models

Call the low-level language models under LLM agents easily.

It contains LLLM and LLLM-Template

## Installation

You can install the reusable `lllm` package directly from this repository:

```bash
pip install git+https://github.com/ChengJunyan1/lllm.git
```

After installation, import it in your project:

```python
import lllm
print(lllm.__version__)
```

## Prompt & Proxy auto-registration

Place an `lllm.toml` file in your project (or set `LLLM_CONFIG=/path/to/lllm.toml`).  
Example (`template/lllm.toml`):

```toml
[prompts]
folders = ["system/agent/prompts"]

[proxies]
folders = ["system/proxy/modules"]
```

When `lllm` is imported it automatically scans these folders, registers every `Prompt` instance, and registers every `BaseProxy` subclass.  
Set `LLLM_AUTO_DISCOVER=0` to skip auto-loading if you prefer to register things manually.

