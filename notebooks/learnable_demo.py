# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "marimo",
#   "openai",
#   "python-dotenv",
#   "pydantic",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv
    from openai import OpenAI
    from pydantic import BaseModel
    from typing import Literal
    from smartfunc import learnable, Pipeline

    load_dotenv(".env", override=True)
    return BaseModel, Literal, OpenAI, Pipeline, learnable, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Learnable Pipelines (Prompt Optimization)

    This notebook shows how to define prompt-based learnable functions and
    optimize them against input/output examples with a metric.

    We'll train a simple classifier to detect whether a sentence about
    \"Python\" is about programming or the animal.
    """)
    return


@app.cell
def _(OpenAI):
    client = OpenAI()
    return (client,)


@app.cell
def _(BaseModel, Literal, Pipeline, client, learnable):
    class TopicLabel(BaseModel):
        label: Literal["PROGRAMMING", "ANIMAL"]

    @learnable(
        client,
        model="gpt-4o-mini",
        prompt=(
            "Classify if the sentence is about programming (Python language) "
            "or the animal. Reply with JSON that matches this schema:\n"
            "{ \"label\": \"PROGRAMMING\" | \"ANIMAL\" }\n"
            "Sentence: {text}"
        ),
        output_model=TopicLabel,
    )
    def classify(text: str):
        pass  # auto-maps inputs into the prompt

    pipeline = Pipeline(classify)
    return (pipeline,)


@app.cell
def _():
    examples = [
        {"input": "I wrote a Python script to automate reports.", "output": "PROGRAMMING"},
        {"input": "The python curled up on the warm rock.", "output": "ANIMAL"},
        {"input": "We deployed a Python service to production.", "output": "PROGRAMMING"},
        {"input": "The python shed its skin overnight.", "output": "ANIMAL"},
        {"input": "Python has great libraries for data analysis.", "output": "PROGRAMMING"},
        {"input": "A python can swallow prey whole.", "output": "ANIMAL"},
    ]

    def metric(example: dict) -> float:
        return 1.0 if example["prediction"].label == example["output"] else 0.0
    return examples, metric


@app.cell
def _(pipeline):
    base_prompt = pipeline.classify.base_prompt
    prompt_before = pipeline.classify.prompt
    return base_prompt, prompt_before


@app.cell
def _(examples, metric, pipeline):
    run_learning = False

    if run_learning:
        pipeline.learn(examples=examples, metric=metric, steps=2, candidates=2)
    return


@app.cell
def _(pipeline):
    result = pipeline("Python lets you build quick prototypes.")
    result.label
    return


@app.cell
def _(base_prompt, mo, pipeline, prompt_before):
    prompt_after = pipeline.classify.prompt

    mo.md(
        f\"\"\"\n**Base prompt**\n\n```\n{base_prompt}\n```\n\n**Prompt before learning**\n\n```\n{prompt_before}\n```\n\n**Prompt after learning**\n\n```\n{prompt_after}\n```\n\"\"\"\n    )
    return prompt_after


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
