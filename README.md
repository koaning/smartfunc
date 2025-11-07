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

This library uses the OpenAI SDK to interact with LLMs. Your function can return either a string (which becomes the prompt) or a list of message dictionaries (for full conversation control). The decorator handles calling the LLM and parsing the response.

The key benefits of this approach:

- **Works with any OpenAI SDK-compatible provider**: Use OpenAI, OpenRouter, or any provider with OpenAI-compatible APIs
- **Full Python control**: Build prompts using Python (no template syntax to learn)
- **Type-safe structured outputs**: Use Pydantic models for validated responses
- **Async support**: Built-in async/await support for concurrent operations
- **Conversation history**: Pass message lists for multi-turn conversations
- **Multimodal support**: Include images, audio, and video via base64 encoding
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

This will return a Pydantic model that might look like this: 

```python
Summary(
    summary='Pikachu is a small, electric-type Pokémon...',
    pros=['Iconic mascot', 'Strong electric attacks', 'Cute appearance'],
    cons=['Weak against ground-type moves', 'Limited evolution options']
)
```

### System Prompts and Parameters

You can confirm anything you like upfront in the client. 

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
    return f"Provide an expert analysis of {pokemon}"
```

### Async Support

If you like working asynchronously, you can use `async_backend` for non-blocking operations. Beware that you may get throttled by the LLM provider if you send too many requests too quickly.

```python
import asyncio
from smartfunc import async_backend
from openai import AsyncOpenAI

client = AsyncOpenAI()

@async_backend(client, model="gpt-4o-mini", response_format=Summary)
async def analyze_async(pokemon: str) -> Summary:
    return f"Describe: {pokemon}"

result = asyncio.run(analyze_async("charizard"))
print(result)
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

### Conversation History

Instead of returning a string, you can return a list of message dictionaries to have full control over the conversation:

```python
@backend(client, model="gpt-4o-mini")
def chat_with_history(user_message: str, conversation_history: list) -> list:
    """Chat with conversation context."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    # Add previous conversation
    messages.extend(conversation_history)

    # Add new user message
    messages.append({"role": "user", "content": user_message})

    return messages

# Use it with conversation history
history = [
    {"role": "user", "content": "What's your name?"},
    {"role": "assistant", "content": "I'm Claude, an AI assistant."},
]

response = chat_with_history("What can you help me with?", history)
print(response)
```

Note: When you return a message list, the `system` parameter in the decorator is ignored.

### Multimodal Content (Images, Audio, Video)

You can include images, audio, or video by passing them as base64-encoded content in your messages:

```python
import base64

@backend(client, model="gpt-4o-mini")
def analyze_image(image_path: str, question: str) -> list:
    """Analyze an image with a question."""
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    },
                },
            ],
        }
    ]

result = analyze_image("photo.jpg", "What's in this image?")
print(result)
```

You can also mix multiple media types:

```python
@backend(client, model="gpt-4o-mini")
def analyze_multiple_media(image1_path: str, image2_path: str) -> list:
    """Compare two images."""
    # Encode images
    with open(image1_path, "rb") as f:
        img1 = base64.b64encode(f.read()).decode("utf-8")
    with open(image2_path, "rb") as f:
        img2 = base64.b64encode(f.read()).decode("utf-8")

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these images:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img1}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img2}"},
                },
            ],
        }
    ]

result = analyze_multiple_media("image1.jpg", "image2.jpg")
```

For audio content:

```python
@backend(client, model="gpt-4o-mini")
def transcribe_audio(audio_path: str) -> list:
    """Transcribe audio content."""
    with open(audio_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio:"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": "wav"  # or "mp3", "flac", etc.
                    },
                },
            ],
        }
    ]
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

<details>
<summary>If you're upgrading from v0.2.0, here are the key changes:</summary>

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

</details>

## Development

Run tests:

```bash
make check
```

Or:

```bash
uv run pytest tests
```
