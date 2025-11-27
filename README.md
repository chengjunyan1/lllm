<div align="center">
  <img src="assets/LLLM-logo.png" alt="LLLM Logo" width="600" style="margin-left:'auto' margin-right:'auto'"/>
  <br>
  <br>
  <h1>Low-Level Language Models (LLLM) </h1>
</div>


An LLM-agentic system framework. It contains LLLM and LLLM-Template. Check the document here: https://junyan.ch/lllm/

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

## Project scaffold CLI

Create a fresh project skeleton by running:

```bash
lllm create --name system
```

This creates a `system/` directory containing:
- `lllm.toml` pointing to prompt/proxy folders
- `config/<name>/default.yaml` with stub settings
- `system/agent/prompts/` and `system/proxy/modules/` with starter files

Edit the generated files and you're ready to build your agent. The command fails if the destination already exists so you never overwrite existing work by accident.

<!-- ## Documentation

Comprehensive documentation (design philosophy, module reference, and build guides) lives under `docs/` and is published via [MkDocs](https://www.mkdocs.org/). Serve it locally with:

```bash
pip install mkdocs
mkdocs serve
```

To generate the static site for GitHub Pages run `mkdocs build`. -->
