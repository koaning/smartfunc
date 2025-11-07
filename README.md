<img src="imgs/logo.png" width="125" height="125" align="right" />

### smartfunc

> Turn functions into LLM-powered endpoints using OpenAI SDK

If you're keen for a demo, you may appreciate this YouTube video:

[![image](https://github.com/user-attachments/assets/8dee073b-e922-43d7-9ce8-def72a868844)
](https://youtu.be/j9jh46R0ryY)

## Installation

```bash
uv pip install smartfunc
```

You'll also need to set up your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

## What is this?

Here is a nice example of what is possible with this library:

```python
from smartfunc import backend
from openai import OpenAI

client = OpenAI()

@backend(client, model="gpt-4o-mini")
def generate_summary(text: str) -> str:
    return f"Generate a summary of the following text: {text}"
```

The `generate_summary` function will now return a string with the summary of the text that you give it.

## How does it work?

This library uses the OpenAI SDK to interact with LLMs. Your function returns a string that becomes the prompt, and the decorator handles calling the LLM and parsing the response.

The key benefits of this approach:

- **Works with any OpenAI SDK-compatible provider**: Use OpenAI, OpenRouter, or any provider with OpenAI-compatible APIs
- **Full Python control**: Build prompts using Python (no template syntax to learn)
- **Type-safe structured outputs**: Use Pydantic models for validated responses
- **Async support**: Built-in async/await support for concurrent operations
- **Simple and focused**: Does one thing well - turn functions into LLM calls

## Features

### Basic Usage

The simplest way to use `smartfunc`:

```python
from smartfunc import backend
from openai import OpenAI

client = OpenAI()

@backend(client, model="gpt-4o-mini")
def write_poem(topic: str) -> str:
    return f"Write a short poem about {topic}"

print(write_poem("summer"))
```

### Structured Outputs

Use Pydantic models to get validated, structured responses:

```python
from smartfunc import backend
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class Summary(BaseModel):
    summary: str
    pros: list[str]
    cons: list[str]

@backend(client, model="gpt-4o-mini", response_format=Summary)
def analyze_pokemon(name: str) -> str:
    return f"Describe the following pokemon: {name}"

result = analyze_pokemon("pikachu")
print(result.summary)
print(result.pros)
print(result.cons)
```

This will return a Pydantic model with structured data:

```python
Summary(
    summary='Pikachu is a small, electric-type Pokémon...',
    pros=['Iconic mascot', 'Strong electric attacks', 'Cute appearance'],
    cons=['Weak against ground-type moves', 'Limited evolution options']
)
```

### System Prompts and Parameters

You can configure system prompts and pass any OpenAI API parameters:

```python
@backend(
    client,
    model="gpt-4o-mini",
    response_format=Summary,
    system="You are a Pokemon expert with 20 years of experience",
    temperature=0.7,
    max_tokens=500
)
def expert_analysis(pokemon: str) -> Summary:
    """Expert Pokemon analysis."""
    return f"Provide an expert analysis of {pokemon}"
```

### Async Support

Use `async_backend` for non-blocking operations:

```python
import asyncio
from smartfunc import async_backend
from openai import AsyncOpenAI

client = AsyncOpenAI()

@async_backend(client, model="gpt-4o-mini", response_format=Summary)
async def analyze_async(pokemon: str) -> Summary:
    """Async Pokemon analysis."""
    return f"Describe: {pokemon}"

result = asyncio.run(analyze_async("charizard"))
print(result)
```

Async is great for processing multiple items concurrently:

```python
async def analyze_many(pokemon_list: list[str]):
    tasks = [analyze_async(p) for p in pokemon_list]
    return await asyncio.gather(*tasks)

results = asyncio.run(analyze_many(["pikachu", "charizard", "mewtwo"]))
```

### Complex Prompt Logic

Since prompts are built with Python, you can use any logic you want:

```python
@backend(client, model="gpt-4o-mini")
def custom_prompt(items: list[str], style: str, include_summary: bool) -> str:
    """Generate with custom logic."""
    prompt = f"Write in {style} style:\n\n"

    for i, item in enumerate(items, 1):
        prompt += f"{i}. {item}\n"

    if include_summary:
        prompt += "\nProvide a brief summary at the end."

    return prompt

result = custom_prompt(
    items=["First point", "Second point", "Third point"],
    style="formal",
    include_summary=True
)
```

### Using OpenRouter

OpenRouter provides access to hundreds of models through an OpenAI-compatible API:

```python
from openai import OpenAI
import os

# OpenRouter client
openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Use Llama via OpenRouter
@backend(openrouter_client, model="meta-llama/llama-3.1-70b", response_format=Summary)
def analyze_with_llama(pokemon: str) -> Summary:
    return f"Analyze {pokemon}"
```

### Reusable Backend Configurations

You can create reusable backend configurations:

```python
from smartfunc import backend
from openai import OpenAI

client = OpenAI()

# Create a configured backend
gpt_mini = lambda **kwargs: backend(
    client,
    model="gpt-4o-mini",
    system="You are a helpful assistant",
    temperature=0.7,
    **kwargs
)

# Use it multiple times
@gpt_mini(response_format=Summary)
def summarize(text: str) -> Summary:
    return f"Summarize: {text}"

@gpt_mini()
def translate(text: str, language: str) -> str:
    return f"Translate '{text}' to {language}"
```

## Migration from v0.2.0

If you're upgrading from v0.2.0, here are the key changes:

### What Changed

1. **Client injection required**: You now pass an OpenAI client instance instead of a model name string
2. **Functions return prompts**: Your function should return a string (the prompt), not use docstrings as templates
3. **`response_format` parameter**: Structured output is specified via `response_format=` instead of return type annotations
4. **No more Jinja2**: Prompts are built with Python, not templates

### Before (v0.2.0)

```python
from smartfunc import backend

@backend("gpt-4o-mini")
def summarize(text: str) -> Summary:
    """Summarize: {{ text }}"""
    pass
```

### After (v1.0.0)

```python
from smartfunc import backend
from openai import OpenAI

client = OpenAI()

@backend(client, model="gpt-4o-mini", response_format=Summary)
def summarize(text: str) -> Summary:
    """This is now actual documentation."""
    return f"Summarize: {text}"
```

### Why the Changes?

- **Better type checking**: The `response_format` parameter doesn't interfere with type checkers
- **More flexibility**: Full Python for prompt generation instead of Jinja2 templates
- **Multi-provider support**: Works with any OpenAI SDK-compatible provider (OpenRouter, etc.)
- **Explicit dependencies**: Client injection makes it clear what's being used
- **Simpler codebase**: Removed magic template parsing

## Development

Run tests:

```bash
make check
```

Or:

```bash
uv run pytest tests
```
