import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import asyncio
    import base64
    import os
    from dotenv import load_dotenv
    from openai import OpenAI, AsyncOpenAI
    from pydantic import BaseModel
    from smartfunc import backend, async_backend, learnable, Pipeline

    load_dotenv(".env", override=True)
    return (
        AsyncOpenAI,
        BaseModel,
        OpenAI,
        Pipeline,
        async_backend,
        asyncio,
        backend,
        base64,
        learnable,
        load_dotenv,
        mo,
        os,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # README Examples

        This notebook mirrors the examples from the README. Toggle the
        `run_*` flags to execute calls.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## What is this?")
    return


@app.cell
def _(OpenAI, backend):
    client = OpenAI()

    @backend(client, model="gpt-4o-mini")
    def generate_summary(text: str) -> str:
        return f"Generate a summary of the following text: {text}"

    return client, generate_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Other Providers")
    return


@app.cell
def _(OpenAI, os):
    other_provider_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    return (other_provider_client,)


@app.cell
def _(BaseModel):
    class Summary(BaseModel):
        summary: str
        pros: list[str]
        cons: list[str]

    return (Summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Basic Usage")
    return


@app.cell
def _(backend, client):
    @backend(client, model="gpt-4o-mini")
    def write_poem(topic: str) -> str:
        return f"Write a short poem about {topic}"

    run_basic = False
    if run_basic:
        write_poem("summer")

    return run_basic, write_poem


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Structured Outputs")
    return


@app.cell
def _(Summary, backend, client):
    @backend(client, model="gpt-4o-mini", response_format=Summary)
    def analyze_pokemon(name: str) -> Summary:
        return f"Describe the following pokemon: {name}"

    run_structured = False
    if run_structured:
        result = analyze_pokemon("pikachu")
        (result.summary, result.pros, result.cons)

    return analyze_pokemon, run_structured


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
Example structured output:

```python
Summary(
    summary='Pikachu is a small, electric-type Pokemon...',
    pros=['Iconic mascot', 'Strong electric attacks', 'Cute appearance'],
    cons=['Weak against ground-type moves', 'Limited evolution options']
)
```
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## System Prompts and Parameters")
    return


@app.cell
def _(Summary, backend, client):
    @backend(
        client,
        model="gpt-4o-mini",
        response_format=Summary,
        system="You are a Pokemon expert with 20 years of experience",
        temperature=0.7,
        max_tokens=500,
    )
    def expert_analysis(pokemon: str) -> Summary:
        return f"Provide an expert analysis of {pokemon}"

    return (expert_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Async Support")
    return


@app.cell
def _(AsyncOpenAI, Summary, async_backend, asyncio):
    async_client = AsyncOpenAI()

    @async_backend(async_client, model="gpt-4o-mini", response_format=Summary)
    async def analyze_async(pokemon: str) -> Summary:
        return f"Describe: {pokemon}"

    run_async = False
    if run_async:
        asyncio.run(analyze_async("charizard"))

    return analyze_async, async_client, run_async


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Complex Prompt Logic")
    return


@app.cell
def _(backend, client):
    @backend(client, model="gpt-4o-mini")
    def custom_prompt(items: list[str], style: str, include_summary: bool) -> str:
        prompt = f"Write in {style} style:\n\n"
        for i, item in enumerate(items, 1):
            prompt += f"{i}. {item}\n"
        if include_summary:
            prompt += "\nProvide a brief summary at the end."
        return prompt

    run_custom = False
    if run_custom:
        custom_prompt(
            items=["First point", "Second point", "Third point"],
            style="formal",
            include_summary=True,
        )

    return custom_prompt, run_custom


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Learnable Pipelines")
    return


@app.cell
def _(Pipeline, client, learnable):
    @learnable(client, model="gpt-4o-mini", prompt="Filter: {x}")
    def filter_func(x: str):
        pass

    @learnable(client, model="gpt-4o-mini", prompt="Rank: {x}")
    def ranker_func(x: str):
        return {"x": x}

    pipeline = Pipeline(filter_func, ranker_func)

    return filter_func, pipeline, ranker_func


@app.cell
def _():
    examples = [
        {"input": "cats", "output": "ranked cats"},
        {"input": "dogs", "output": "ranked dogs"},
    ]

    def metric(example: dict) -> float:
        return 1.0 if example["prediction"] == example["output"] else 0.0

    return examples, metric


@app.cell
def _(examples, metric, pipeline):
    run_learning = False
    if run_learning:
        pipeline.learn(examples=examples, metric=metric, steps=3, candidates=2)

    return (run_learning,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Conversation History")
    return


@app.cell
def _(backend, client):
    @backend(client, model="gpt-4o-mini")
    def chat_with_history(user_message: str, conversation_history: list) -> list:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    run_chat = False
    if run_chat:
        history = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "I'm Claude, an AI assistant."},
        ]
        chat_with_history("What can you help me with?", history)

    return chat_with_history, run_chat


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Multimodal Content")
    return


@app.cell
def _(backend, base64, client):
    @backend(client, model="gpt-4o-mini")
    def analyze_image(image_path: str, question: str) -> list:
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

    @backend(client, model="gpt-4o-mini")
    def analyze_multiple_media(image1_path: str, image2_path: str) -> list:
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

    @backend(client, model="gpt-4o-mini")
    def transcribe_audio(audio_path: str) -> list:
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
                            "format": "wav",
                        },
                    },
                ],
            }
        ]

    return analyze_image, analyze_multiple_media, transcribe_audio


@app.cell(hide_code=True)
def _(mo):
    mo.md("## OpenRouter Example")
    return


@app.cell
def _(OpenAI, Summary, backend, os):
    openrouter_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    @backend(openrouter_client, model="meta-llama/llama-3.1-70b", response_format=Summary)
    def analyze_with_llama(pokemon: str) -> Summary:
        return f"Analyze {pokemon}"

    return analyze_with_llama, openrouter_client


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Reusable Backend Configurations")
    return


@app.cell
def _(Summary, backend, client):
    gpt_mini = lambda **kwargs: backend(
        client,
        model="gpt-4o-mini",
        system="You are a helpful assistant",
        temperature=0.7,
        **kwargs,
    )

    @gpt_mini(response_format=Summary)
    def summarize(text: str) -> Summary:
        return f"Summarize: {text}"

    @gpt_mini()
    def translate(text: str, language: str) -> str:
        return f"Translate '{text}' to {language}"

    return gpt_mini, summarize, translate


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Migration from v0.2.0

### Before (v0.2.0)

```python
from smartfunc import backend

@backend("gpt-4o-mini")
def summarize(text: str) -> Summary:
    '''Summarize: {{ text }}'''
    pass
```

### After (v1.0.0)

```python
from smartfunc import backend
from openai import OpenAI

client = OpenAI()

@backend(client, model="gpt-4o-mini", response_format=Summary)
def summarize(text: str) -> Summary:
    '''This is now actual documentation.'''
    return f"Summarize: {text}"
```
"""
    )
    return


if __name__ == "__main__":
    app.run()
